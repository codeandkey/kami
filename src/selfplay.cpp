#include "selfplay.h"
#include "env.h"
#include "mcts.h"
#include "evaluate.h"
#include "options.h"

#include <iostream>
#include <stdexcept>
#include <cmath>

using namespace kami;
using namespace std;

Selfplay::Selfplay(NN* model) :
    model(model),
    ibatch(options::getInt("selfplay_batch", 16)),
    nodes(options::getInt("selfplay_nodes", 512)),
    replay_buffer(8 * 8 * NFEATURES, PSIZE, options::getInt("replaybuffer_size", 512)) {}

void Selfplay::start()
{
    status.code(RUNNING);

    int n_inference = options::getInt("inference_threads", 1);

    for (int i = 0; i < n_inference; ++i)
    {
        partial_trajectories.emplace_back(0);
        inference.push_back(thread(&Selfplay::inference_main, this, i));
    }

    for (int i = 0; i < options::getInt("training_threads", 1); ++i)
        training.push_back(thread(&Selfplay::training_main, this, i));
}

void Selfplay::stop()
{
    if (status.code() != RUNNING)
        throw runtime_error("stop() called when not running");

    status.code(WAITING);

    // Wait for threads
    for (auto& t : inference)
        t.join();

    inference.clear();

    for (auto& t : training)
        t.join();

    training.clear();

    status.code(STOPPED);
}

void Selfplay::inference_main(int id) { try {
    cout << "Starting inference thread: " << id << endl;

    bool flush_old_trees = options::getInt("flush_old_trees", 1);

    struct T {
        T(float* i, float* l, float* m) {
            inputs = new float[8 * 8 * NFEATURES];
            lmm = new float[PSIZE];
            mcts = new float[PSIZE];

            memcpy(inputs, i, sizeof(float) * 8 * 8 * NFEATURES);
            memcpy(lmm, l, sizeof(float) * PSIZE);
            memcpy(mcts, m, sizeof(float) * PSIZE);
        }

        ~T() {
            delete[] inputs;
            delete[] lmm;
            delete[] mcts;
        }

        float* inputs = nullptr, *lmm = nullptr, *mcts = nullptr, result;
    };
    
    // Spin up environments
    MCTS trees[ibatch];
    vector<vector<T*>> trajectories;
    vector<int> source_generation;

    for (int i = 0; i < ibatch; ++i)
    {
        trajectories.emplace_back();
        source_generation.push_back(model->get_generation());
    }

    float* batch = new float[ibatch * 8 * 8 * NFEATURES];
    float* lmm = new float[ibatch * PSIZE];
    float* inf_value = new float[ibatch];
    float* inf_policy = new float[ibatch * PSIZE];

    int partials = 0;

    while (status.code() == RUNNING)
    {
        // Build next batch
        for (int i = 0; i < ibatch; ++i)
        {
            // Check if tree is out of date and needs replacing
            if (flush_old_trees && source_generation[i] < model->get_generation())
            {
                // Replace environment and start again
                trees[i].reset();

                for (T*& t : trajectories[i])
                    delete t;

                partials -= trajectories[i].size();
                trajectories[i].clear();
                source_generation[i] = model->get_generation();
            }

            // Push up to node limit, or next observation
            while (trees[i].n() < nodes && !trees[i].select(batch + i * 8 * 8 * NFEATURES, lmm + i * PSIZE));

            // If not ready, this observation is done
            if (trees[i].n() < nodes) continue;

            // Otherwise, save this trajectory and perform the action

            // Reuse batch space, it will be overwritten anyway
            trees[i].get_env().observe(batch + i * 8 * 8 * NFEATURES);
            trees[i].get_env().lmm(lmm + i * PSIZE);

            float mcts[PSIZE];
            trees[i].snapshot(mcts);

            ++partials;
            trajectories[i].push_back(new T(batch + i * 8 * 8 * NFEATURES, lmm + i * PSIZE, mcts));

            float alpha = ALPHA_FINAL;

            if (trees[i].get_env().ply() < ALPHA_CUTOFF)
                alpha = pow(ALPHA_DECAY, trees[i].get_env().ply()) * ALPHA_INITIAL;

            int picked = trees[i].pick(alpha);

            trees[i].push(picked);

            // Check terminal state
            float value;

            if (trees[i].get_env().terminal(&value))
            {
                if (wants_pgn.exchange(false))
                {
                    ret_pgn = trees[i].get_env().pgn();
                    wants_pgn = false;
                }

                // Replace environment and reobserve
                trees[i].reset();

                for (auto& t : trajectories[i])
                {
                    t->result = value;
                    replay_buffer.add(t->inputs, t->lmm, t->mcts, t->result);
                    delete t;
                }

                partials -= trajectories[i].size();
                trajectories[i].clear();
            }

            // Try again on new env
            --i;
            continue;
        }

        // Inference
        model->infer(batch, lmm, ibatch, inf_policy, inf_value);

        // Expansion
        for (int i = 0; i < ibatch; ++i)
            trees[i].expand(inf_policy + i * PSIZE, inf_value[i]);

        // Update partial trajectories
        auto pt = partial_trajectories.begin();
        advance(pt, id);
        *pt = partials;
    }

    delete[] batch;
    delete[] inf_value;
    delete[] inf_policy;

    cout << "Terminating inference thread: " << id << endl;
} catch (exception& e) {
    cerr << "Inference thread " << id << " failed: " << e.what() << endl;
}}

void Selfplay::training_main(int id)
{
    cout << "Starting training thread " << id << endl;

    string modelpath = options::getStr("model_path", "/tmp/model.pt");

    long target_count = replay_buffer.size(), target_from = 0;
    int target_incr = replay_buffer.size() * options::getInt("rpb_train_pct", 40) / 100;
    int trajectories = replay_buffer.size() * options::getInt("training_sample_pct", 60) / 100;

    float* inputs = new float[trajectories * 8 * 8 * NFEATURES];
    float* lmm = new float[trajectories * PSIZE];
    float* mcts = new float[trajectories * PSIZE];
    float* results = new float[trajectories];

    // Wait for total trajectory target
    while (status.code() == RUNNING)
    {
        // Check if target percentage reached, also grab the current generation
        if (replay_buffer.count() < target_count)
        {
            if (!id)
            {
                cout << "Gen " << model->get_generation() << " RPB " << 100 * (replay_buffer.count() - target_from) / (target_count - target_from) << "% [" << replay_buffer.count() - target_from << " / " << target_count - target_from << "] | Partials: ";

                auto ct = partial_trajectories.begin();

                for (int inf = 0; inf < inference.size(); ++inf)
                {
                    cout << " Inf " << inf << ": " << *ct;
                    ++ct;
                }

                cout << endl;
            }

            this_thread::sleep_for(chrono::milliseconds(1000));
            continue;
        }

        // Ready to train
        cout << "Trainer " << id << ": training generation " << model->get_generation() << " with " << trajectories << " trajectories sampled from last " << replay_buffer.size() << endl;

        // Clone the current model
        NN cmodel(model);

        // Train new model
        replay_buffer.select_batch(inputs, lmm, mcts, results, trajectories);
        cmodel.train(trajectories, inputs, lmm, mcts, results);

        // Evaluate new model
        if (eval(model, &cmodel))
        {
            // Save the current model
            cmodel.write(modelpath);
            model->read(modelpath);

            cout << "Candidate accepted: using new generation " << model->get_generation() << endl;

            if (options::getInt("flush_old_rpb", 1))
                replay_buffer.clear();

            target_count = max((long) replay_buffer.size(), replay_buffer.count() + (long) target_incr);
            target_from = replay_buffer.count();
            
            continue;
        } else
        {
            cout << "Candidate rejected: generation remains " << model->get_generation() << endl;
        }

        target_from = replay_buffer.count();
        target_count += target_incr;
    }

    cout << "Stopping training thread" << endl;
}
