#include "selfplay.h"
#include "env.h"
#include "mcts.h"

#include <iostream>
#include <stdexcept>

using namespace kami;
using namespace std;

Selfplay::Selfplay(NN* model, int ibatch, int nodes) :
    model(model),
    ibatch(ibatch),
    nodes(nodes),
    replay_buffer(8 * 8 * NFEATURES, PSIZE, 2048) {}

void Selfplay::start(int n_inference)
{
    status.code(RUNNING);

    for (int i = 0; i < n_inference; ++i)
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
    std::vector<MCTS> trees;
    vector<vector<T*>> trajectories;

    for (int i = 0; i < ibatch; ++i)
    {
        trees.emplace_back();
        trajectories.emplace_back();
    }

    float* batch = new float[ibatch * 8 * 8 * NFEATURES];
    float* lmm = new float[ibatch * PSIZE];
    float* inf_value = new float[ibatch];
    float* inf_policy = new float[ibatch * PSIZE];

    cout << "INFERENCE initialized batch buffers" << endl;

    while (status.code() == RUNNING)
    {
        // Build next batch
        for (int i = 0; i < ibatch; ++i)
        {
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

            int picked = trees[i].pick();

            trees[i].push(picked);

            // Check terminal state
            float value;

            if (trees[i].get_env().terminal(&value))
            {
                // Replace environment and reobserve
                trees[i].reset();
                cout << "Recorded " << trajectories[i].size() << " experiences from tree " << i << endl;

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

            // Get next observation from new env
            //while (!trees[i].select(batch + i * 8 * 8 * NFEATURES, lmm + i * PSIZE));
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

    long last_total = 0, target_count = replay_buffer.size();

    float* inputs = new float[TRAIN_BATCHSIZE * 8 * 8 * NFEATURES];
    float* lmm = new float[TRAIN_BATCHSIZE * PSIZE];
    float* mcts = new float[TRAIN_BATCHSIZE * PSIZE];
    float* results = new float[TRAIN_BATCHSIZE];

    // Wait for total trajectory target
    while (status.code() == RUNNING)
    {
        // Check if replay buffer is filled and target percentage reached
        if (replay_buffer.count() < replay_buffer.size())
        {
            this_thread::sleep_for(chrono::milliseconds(100));
            continue;
        }

        // Check if target percentage reached
        if (replay_buffer.count() < target_count)
        {
            this_thread::sleep_for(chrono::milliseconds(100));
            continue;
        }

        // Ready to train
        cout << "Starting training step at " << replay_buffer.count() << " trajectories" << endl;

        replay_buffer.select_batch(inputs, lmm, mcts, results, TRAIN_BATCHSIZE);
        model->train(TRAIN_BATCHSIZE, inputs, lmm, mcts, results);

        target_count += replay_buffer.size() * TRAIN_AFTER_PCT / 100;
    }

    cout << "Stopping training thread" << endl;
}
