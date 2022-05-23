#include "../src/nn/nn.h"
#include "../src/env.h"

#define TESTSIZE 5000 // observations per batch test

using namespace kami;
using namespace std;

int main() {
    float* inp = new float[128 * 8 * 8 * NFEATURES];

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
