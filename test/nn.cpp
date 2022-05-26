#include "../src/nn/nn.h"
#include "../src/env.h"

#define TESTSIZE 1000

using namespace kami;
using namespace std;

int main() {
    float* inp = new float[128 * 8 * 8 * NFEATURES];
    float* lmm = new float[128 * PSIZE];

    for (int j = 0; j < 128 * PSIZE; ++j)
        lmm[j] = ((float) rand() / (float) RAND_MAX) > 0.5f ? 1.0f : 0.0f;

    NN net(8, 8, NFEATURES, PSIZE, true);

    for (int i = 8; i <= 128; i += 8)
    {
        clock_t start = clock();

        float policy[i * PSIZE];
        float value[i];

        for (int b = 0; b < TESTSIZE / i; ++b)
        {
            for (int j = 0; j < i * 8 * 8 * NFEATURES; ++j)
                inp[j] = (float) rand() / (float) RAND_MAX;

            net.infer(inp, lmm, i, policy, value);
        }

        cout << "batch size " << i << " : " << (TESTSIZE * CLOCKS_PER_SEC) / (clock() - start) << " pred/s\n";
    }

    delete[] inp;
    delete[] lmm;

    return 0;
}
