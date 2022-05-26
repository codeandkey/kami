#include "../src/nn/nn.h"
#include "../src/env.h"

#include <torch/cuda.h>

#define TESTSIZE 4096 // observations per batch test

using namespace kami;
using namespace std;

int main() {
    float* inp = new float[128 * 8 * 8 * NFEATURES];
    float* lmm = new float[128 * PSIZE];

    for (int j = 0; j < 128 * PSIZE; ++j)
        lmm[j] = ((float) rand() / (float) RAND_MAX) > 0.5f ? 1.0f : 0.0f;

    NN net(8, 8, NFEATURES, PSIZE);

    if (!net.isCUDA())
    {
        cout << "Couldn't initialize model in CUDA mode, aborting\n";
        return 0;
    }

    for (int f = 0; f < 5; ++f)
    {
        clock_t start = clock(), timer = start;
        int bsize = 8 << f; 
        float policy[bsize * PSIZE];
        float value[bsize];

        for (int batch = 1; batch <= TESTSIZE / bsize; ++batch)
        {
            for (int i = 0; i < bsize * 8 * 8 * NFEATURES; ++i)
                inp[i] = (float) rand() / (float) RAND_MAX;

            net.infer(inp, lmm, bsize, policy, value);
        }

        cout << "batch size " << bsize << " : " << (TESTSIZE * CLOCKS_PER_SEC) / (clock() - start) << " pred/s" << endl;
    }

    delete[] inp;
    delete[] lmm;

    return 0;
}
