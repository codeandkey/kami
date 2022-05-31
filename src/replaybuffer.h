#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <mutex>

namespace kami {

class ReplayBuffer {
    public:
        ReplayBuffer(
            int obsize,
            int psize,
            int bufsize) : 
            obsize(obsize),
            psize(psize),
            bufsize(bufsize)
        {
            input_buffer = new float[obsize * bufsize];
            lmm_buffer = new float[bufsize * psize];
            mcts_buffer = new float[bufsize * psize];
            result_buffer = new float[bufsize];
        }

        ~ReplayBuffer() {
            delete[] input_buffer;
            delete[] lmm_buffer;
            delete[] result_buffer;
            delete[] mcts_buffer;
        }

        void add(float* input, float* lmm, float* mcts, float result)
        {
            std::lock_guard<std::mutex> lock(buffer_mut);

            memcpy(
                input_buffer + write_index * obsize,
                input,
                sizeof(float) * obsize
            );

            memcpy(
                lmm_buffer + write_index * psize,
                lmm,
                sizeof(float) * psize
            );

            memcpy(
                mcts_buffer + write_index * psize,
                mcts,
                sizeof(float) * psize
            );

            result_buffer[write_index++] = result;
            write_index %= bufsize;

            ++total;
        }

        int size() { return bufsize; }
        long count() { return total; }

        void select_batch(float* dst_input, float* dst_lmm, float* dst_mcts, float* dst_result, int n)
        {
            std::lock_guard<std::mutex> lock(buffer_mut);

            // Don't worry about duplicates. Make n selections and copy into the batch.
            for (int i = 0; i < n; ++i)
            {
                int source = rand() % bufsize;

                memcpy(
                    dst_input + i * obsize,
                    input_buffer + source * obsize,
                    sizeof(float) * obsize
                );

                memcpy(
                    dst_lmm + i * psize,
                    lmm_buffer + source * psize,
                    sizeof(float) * psize
                );

                memcpy(
                    dst_mcts + i * psize,
                    mcts_buffer + source * psize,
                    sizeof(float) * psize
                );

                dst_result[i] = result_buffer[source];
            }
        }

    private:
        int obsize, psize, bufsize;
        std::mutex buffer_mut;
        float* input_buffer, *lmm_buffer, *result_buffer, *mcts_buffer;
        int write_index = 0;
        long total = 0;
}; // class ReplayBuffer
} // namespace kami