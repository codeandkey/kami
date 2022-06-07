#include "evaluate.h"
#include "env.h"
#include "mcts.h"

#include <iostream>

using namespace kami;

bool kami::eval(NN* current_model, NN* candidate_model)
{
    // Pick turns
    int candidate_turns[EVAL_BATCH];

    // Each model plays first half 1, second half -1
    for (int i = 0; i < EVAL_BATCH; ++i)
        candidate_turns[i] = (rand() % 2) * 2 - 1;

    // Input buffers
    float* cur_inputs = new float[EVAL_BATCH * 8 * 8 * NFEATURES];
    float* cur_lmm = new float[EVAL_BATCH * PSIZE];

    // Input buffers (candidate)
    float* cd_inputs = new float[EVAL_BATCH * 8 * 8 * NFEATURES];
    float* cd_lmm = new float[EVAL_BATCH * PSIZE];

    // Trees
    MCTS trees[EVAL_BATCH];

    // P/V
    float policy[EVAL_BATCH * PSIZE];
    float value[EVAL_BATCH];
    float tvalue;

    // Infer tags (game target)
    int cur_targets[EVAL_BATCH];
    int cd_targets[EVAL_BATCH];

    float score = 0.0f; // Score
    int games = 0; // Games played
    
    std::cout << "Starting evaluation of model generation " << candidate_model->generation << " over " << EVAL_RUNS << " games" << std::endl;
    
    // Start playing games
    while (games < EVAL_RUNS)
    {
        int cur_batch_size = 0;
        int cd_batch_size = 0;

        // Build batches
        for (int i = 0; i < sizeof(trees) / sizeof(trees[0]); ++i)
        {
            float* inputs = trees[i].get_env().turn() == candidate_turns[i] ? cd_inputs : cur_inputs;
            float* lmm = trees[i].get_env().turn() == candidate_turns[i] ? cd_lmm : cur_lmm;

            int boff = trees[i].get_env().turn() == candidate_turns[i] ? cd_batch_size : cur_batch_size;

            // Push up to node limit, or next observation
            while (trees[i].n() < EVAL_NODES && !trees[i].select(inputs + boff * 8 * 8 * NFEATURES, lmm + boff * PSIZE));

            // If not ready, this observation is done, we pass it to the model
            if (trees[i].n() < EVAL_NODES)
            {
                if (trees[i].get_env().turn() == candidate_turns[i])
                    cd_targets[cd_batch_size++] = i;
                else
                    cur_targets[cur_batch_size++] = i;

                continue;
            }


            // Make action
            trees[i].push(trees[i].pick());

            if (trees[i].get_env().terminal(&tvalue))
            {
                // Check result
                score += tvalue * candidate_turns[i] / 2.0f + 0.5f;
                games++;

                std::cout << "Game " << games << " of " << EVAL_RUNS << " [" << tvalue * candidate_turns[i] << "]: score " << (int) (score * 100 / games) << "%" << std::endl;

                // Reset model POV to match this model
                trees[i].reset();
                candidate_turns[i] = (inputs == cd_inputs) ? 1.0f : -1.0f;
            }

            // Try again
            --i;

            continue;
        }

        // Batch inference
        if (cur_batch_size)
        {
            current_model->infer(cur_inputs, cur_lmm, cur_batch_size, policy, value);

            for (int i = 0; i < cur_batch_size; ++i)
                trees[cur_targets[i]].expand(policy + i * PSIZE, value[i]);
        }

        // Batch inference (on candidate)
        if (cd_batch_size)
        {
            candidate_model->infer(cd_inputs, cd_lmm, cd_batch_size, policy, value);

            for (int i = 0; i < cd_batch_size; ++i)
                trees[cd_targets[i]].expand(policy + i * PSIZE, value[i]);
        }
    }

    delete[] cur_inputs;
    delete[] cur_lmm;
    delete[] cd_inputs;
    delete[] cd_lmm;

    std::cout << "Finished evaluating: score " << (int) (score * 100 / games) << "%, target " << (int) (EVAL_CRIT * 100) << std::endl;

    return score >= EVAL_CRIT * games;
}
