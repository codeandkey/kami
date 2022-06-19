#include "evaluate.h"
#include "env.h"
#include "mcts.h"
#include "options.h"

#include <iostream>

using namespace kami;

bool kami::eval(NN* current_model, NN* candidate_model)
{
    int ebatch = options::getInt("evaluate_batch");
    int egames = options::getInt("evaluate_games");
    int enodes = options::getInt("evaluate_nodes");
    int etarget = options::getInt("evaluate_target_pct");

    // Pick turns
    int candidate_turns[ebatch];

    // Each model plays first half 1, second half -1
    for (int i = 0; i < ebatch; ++i)
        candidate_turns[i] = (rand() % 2) * 2 - 1;

    // Input buffers
    float* cur_inputs = new float[ebatch * 8 * 8 * NFEATURES];

    // Input buffers (candidate)
    float* cd_inputs = new float[ebatch * 8 * 8 * NFEATURES];

    // Trees
    MCTS trees[ebatch];

    // P/V
    float policy[ebatch * PSIZE];
    float value[ebatch];
    float tvalue;

    // Infer tags (game target)
    int cur_targets[ebatch];
    int cd_targets[ebatch];

    float score = 0.0f; // Score
    int games = 0; // Games played
    
    std::cout << "Starting evaluation of model generation " << candidate_model->get_generation() << " over " << egames << " games" << std::endl;
    
    // Start playing games
    while (games < egames)
    {
        int cur_batch_size = 0;
        int cd_batch_size = 0;

        // Check if model has already been updated
        if (current_model->get_generation() >= candidate_model->get_generation())
        {
            std::cout << "Model was updated during evaluation, skipping!" << std::endl;
            delete[] cur_inputs;
            delete[] cd_inputs;
            return false;
        }

        // Build batches
        for (int i = 0; i < sizeof(trees) / sizeof(trees[0]); ++i)
        {
            float* inputs = trees[i].get_env().turn() == candidate_turns[i] ? cd_inputs : cur_inputs;

            int boff = trees[i].get_env().turn() == candidate_turns[i] ? cd_batch_size : cur_batch_size;

            // Push up to node limit, or next observation
            while (trees[i].n() < enodes && !trees[i].select(inputs + boff * 8 * 8 * NFEATURES));

            // If not ready, this observation is done, we pass it to the model
            if (trees[i].n() < enodes)
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

                std::cout << "Game " << games << " of " << egames << " [" << tvalue * candidate_turns[i] << "]: score " << (int) (score * 100 / games) << "%" << std::endl;

                // Reset model POV to match this model
                trees[i].reset();
                candidate_turns[i] = (inputs == cd_inputs) ? 1.0f : -1.0f;

                float target_score = (egames * etarget) / 100;

                // Pass or fail early if possible
                if ((score + (egames - games)) < target_score)
                {
                    delete[] cur_inputs;
                    delete[] cd_inputs;
                    std::cout << "Aborting evaluation, score is too low" << std::endl;
                    return false;
                }

                if (score >= target_score)
                {
                    delete[] cur_inputs;
                    delete[] cd_inputs;
                    std::cout << "Passing evaluation early, score is high enough" << std::endl;
                    return true;
                }
            }

            // Try again
            --i;

            continue;
        }

        // Batch inference
        if (cur_batch_size)
        {
            current_model->infer(cur_inputs, cur_batch_size, policy, value);

            for (int i = 0; i < cur_batch_size; ++i)
                trees[cur_targets[i]].expand(policy + i * PSIZE, value[i], true);
        }

        // Batch inference (on candidate)
        if (cd_batch_size)
        {
            candidate_model->infer(cd_inputs, cd_batch_size, policy, value);

            for (int i = 0; i < cd_batch_size; ++i)
                trees[cd_targets[i]].expand(policy + i * PSIZE, value[i], true);
        }
    }

    delete[] cur_inputs;
    delete[] cd_inputs;

    std::cout << "Finished evaluating: score " << (int) (score * 100 / games) << "%, target " << etarget << std::endl;

    return score * 100 / games >= etarget;
}
