#include "../src/nn/nn.h"
#include "../src/env.h"

#define TESTSIZE 4096 // observations per batch test

using namespace kami;
using namespace std;

int main() {
    float* inp = new float[128 * 8 * 8 * NFEATURES];

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

            net.infer(inp, bsize, policy, value);
        }

        cout << "batch size " << bsize << " : " << (TESTSIZE * CLOCKS_PER_SEC) / (clock() - start) << " pred/s" << endl;
    }

    delete[] inp;

    return 0;
}
