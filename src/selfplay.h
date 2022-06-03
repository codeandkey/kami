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

// Train the network after <N>% of the replay buffer has been replaced
constexpr int TRAIN_AFTER_PCT = 40;

class Selfplay {
    public:
        Selfplay(NN* model, int ibatch = 16, int nodes = 1024);

        /**
         * Start the training loop.
         */
        void start(int n_inference = 1);

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
            std::string _message = "Constructed";
            StatusCode _code = STOPPED;
            std::mutex _lock;

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
    private:
        std::vector<std::thread> inference;
        std::thread training;

        NN* model;

        ReplayBuffer replay_buffer;

        int ibatch;
        int nodes;

        void inference_main(int id);
        void training_main();

}; // class Selfplay
} // namespace kami
