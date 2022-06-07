#pragma once

#include "nn/nn.h"
#include "replaybuffer.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

/** Reporting
 *
 * Status structure:
 * String status from training
 * Status enum
 */

namespace kami {

// MCTS alpha parameter curve
constexpr float ALPHA_INITIAL = 1.0f;
constexpr float ALPHA_DECAY = 0.06f;
constexpr int ALPHA_CUTOFF = 15;
constexpr float ALPHA_FINAL = 0.0f;

// Path to model
static const char* MODEL_PATH = "/tmp/model.pt";

class Selfplay {
    public:
        Selfplay(NN* model);

        /**
         * Start the training loop.
         */
        void start();

        /**
         * Stops the training loop and cleans up resources.
         */
        void stop();

        // Status reportingl->infer
        enum StatusCode {
            STOPPED,
            RUNNING,
            WAITING,
        };

        struct Status {
            StatusCode _code = STOPPED;
            std::mutex _lock;
            std::string _message;

            std::string message(std::string text = "") {
                std::lock_guard<std::mutex> lock(_lock);

                if (!text.size())
                    return text;

                return _message = text;            
            }

            StatusCode code(int newcode = -1) {
                std::lock_guard<std::mutex> lock(_lock);

                if (newcode < 0)
                    return _code;

                return _code = StatusCode(newcode);
            } 

        };

        Status status;
        ReplayBuffer& get_rbuf() { return replay_buffer; }

        std::string get_next_pgn() {
            wants_pgn = true;

            while (wants_pgn) 
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

            return ret_pgn;
        }

    private:
        std::vector<std::thread> inference;
        std::thread training;

        NN* model;

        ReplayBuffer replay_buffer;

        int ibatch;
        int nodes;

        std::atomic<bool> wants_pgn;
        std::string ret_pgn;

        void inference_main(int id);
        void training_main();

}; // class Selfplay
} // namespace kami
