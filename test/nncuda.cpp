#include "../src/nn/nn.h"
#include "../src/env.h"

using namespace kami;
using namespace std;

int main() {
    float* inp = new float[128 * 8 * 8 * NFEATURES];
    long bcount = 0;

    if (!torch::cuda::is_available())
    {
        cout << "Torch reports CUDA not available!" << endl;
        return 0;
    }

    cout << "Torch reports " << torch::cuda::device_count() << " CUDA devices" << endl;

    torch::Device dev(torch::kCUDA, 0);

    cout << "CUDA device: " << dev << endl;

    NN net(8, 8, NFEATURES, PSIZE, dev);

    for (int f = 0; f < 4; ++f)
    {
        int i = 8 << f;

        clock_t start = clock();

        float policy[i * PSIZE];
        float value[i];

        for (int b = 0; b < 5000; ++b)
        {
            for (int j = 0; j < i * 8 * 8 * NFEATURES; ++j)
                inp[j] = (float) rand() / (float) RAND_MAX;

            net.infer(inp, i, policy, value);
        }

        bcount = i * 5000;
        cout << "batch size " << i << " : " << (bcount * CLOCKS_PER_SEC) / (clock() - start) << " pred/s\n";
    }

    delete[] inp;
    return 0;
}
