#include "selfplay.h"
#include "env.h"
#include "mcts.h"
#include "evaluate.h"
#include "options.h"

#include <iostream>
#include <stdexcept>

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

    for (int i = 0; i < options::getInt("inference_threads", 1); ++i)
        inference.push_back(thread(&Selfplay::inference_main, this, i));

    training = thread(&Selfplay::training_main, this);
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
    training.join();

    status.code(STOPPED);
}

void Selfplay::inference_main(int id) { try {
    cout << "Starting inference thread: " << id << endl;

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
        source_generation.push_back(model->generation);
    }

    float* batch = new float[ibatch * 8 * 8 * NFEATURES];
    float* lmm = new float[ibatch * PSIZE];
    float* inf_value = new float[ibatch];
    float* inf_policy = new float[ibatch * PSIZE];

    while (status.code() == RUNNING)
    {
        // Build next batch
        for (int i = 0; i < ibatch; ++i)
        {
            // Check if tree is out of date and needs replacing
            if (source_generation[i] < model->generation)
            {
                // Replace environment and start again
                trees[i].reset();

                for (T*& t : trajectories[i])
                    delete t;

                trajectories[i].clear();
                source_generation[i] = model->generation;
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
    }

    delete[] batch;
    delete[] inf_value;
    delete[] inf_policy;

    cout << "Terminating inference thread: " << id << endl;
} catch (exception& e) {
    cerr << "INFERENCE thread failed: " << status.message() << endl;
}}

void Selfplay::training_main()
{
    cout << "Starting training thread" << endl;

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
        // Check if target percentage reached
        if (replay_buffer.count() < target_count)
        {
            cout << "Gen " << model->generation << " RPB " << 100 * (replay_buffer.count() - target_from) / (target_count - target_from) << "% [" << replay_buffer.count() - target_from << " / " << target_count - target_from << "]" << endl;
            cout.flush();
            this_thread::sleep_for(chrono::milliseconds(1000));
            continue;
        }

        // Ready to train
        cout << "Gen " << model->generation << " Starting training step with " << trajectories << " trajectories sampled from last " << replay_buffer.size() << endl;

        // Save the current model
        model->write(modelpath);

        // Generate candidate out-of-place
        NN cmodel(8, 8, NFEATURES, PSIZE, false, model->generation);
        cmodel.read(modelpath);

        // Train new model
        replay_buffer.select_batch(inputs, lmm, mcts, results, trajectories);
        cmodel.train(trajectories, inputs, lmm, mcts, results);

        // Evaluate new model
        if (eval(model, &cmodel))
        {
            // Save the current model
            cmodel.write(modelpath);
            model->read(modelpath);
            model->generation += 1;

            cout << "Candidate accepted: using new generation " << model->generation << endl;

            replay_buffer.clear();
            target_count = replay_buffer.size();
            target_from = 0;
            
            continue;
        } else
        {
            cout << "Candidate rejected: generation remains " << model->generation << endl;
            model->read(modelpath);
        }

        target_from = replay_buffer.count();
        target_count += target_incr;
    }

    cout << "Stopping training thread" << endl;
}
