#include "../src/nn/nn.h"
#include "../src/env.h"

using namespace kami;
using namespace std;

int main() {
    float* inp = new float[64 * 8 * 8 * NFEATURES];
    float policy[PSIZE], value;

    long bcount = 0;

    for (int i = 0; i < 8 * 8 * NFEATURES; ++i)
        inp[i] = 0.1f;

    NN net(8, 8, NFEATURES, PSIZE);

    net.to(torch::kCUDA);

    for (int i = 8; i <= 64; i += 8)
    {
        clock_t start = clock();

        for (int b = 0; b < 500; ++b)
            net.infer(inp, i, policy, &value);

        bcount = i * 500;
        cout << "batch size " << i << " : " << (bcount * CLOCKS_PER_SEC) / (clock() - start) << " infer/second\n";
    }

    return 0;
}
