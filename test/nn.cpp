#include "../src/nn/nn.h"
#include "../src/env.h"

using namespace kami;
using namespace std;

int main() {
    float* inp = new float[128 * 8 * 8 * NFEATURES];
    long bcount = 0;

    NN net(8, 8, NFEATURES, PSIZE, torch::kCPU);

    for (int i = 8; i <= 128; i += 8)
    {
        clock_t start = clock();

        float policy[i * PSIZE];
        float value[i];

        for (int b = 0; b < 500; ++b)
        {
            for (int j = 0; j < i * 8 * 8 * NFEATURES; ++j)
                inp[j] = (float) rand() / (float) RAND_MAX;

            net.infer(inp, i, policy, value);
        }

        bcount = i * 500;
        cout << "batch size " << i << " : " << (bcount * CLOCKS_PER_SEC) / (clock() - start) << " pred/s\n";
    }

    delete[] inp;
    return 0;
}
