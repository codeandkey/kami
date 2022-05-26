#include "../src/nn/nn.h"
#include "../src/env.h"

#include <torch/cuda.h>

using namespace kami;
using namespace std;

int main() {
    NN net(8, 8, NFEATURES, PSIZE);

    float policy[PSIZE], value, inp[8 * 8 * NFEATURES];

    float lmm[PSIZE];

    for (int j = 0; j < 128 * PSIZE; ++j)
        lmm[j] = ((float) rand() / (float) RAND_MAX) > 0.5f ? 1.0f : 0.0f;

    for (int i = 0; i < 8 * 8 * NFEATURES; ++i)
        inp[i] = (float) rand() / (float) RAND_MAX;

    net.infer(inp, lmm, 1, policy, &value);

    float policy2[PSIZE], value2;

    net.write("__nndisk_TESTMODEL.pt");
    net.read("__nndisk_TESTMODEL.pt");

    net.infer(inp, lmm, 1, policy2, &value2);

    for (int i = 0; i < PSIZE; ++i)
        if (policy[i] != policy2[i])
            cerr << "policy mismatch at " << i << ": " << policy[i] << " into " << policy2[i] << endl;

    if (value != value2)
        cerr << "value mismatch: " << value << " into " << value2 << endl;

    return 0;
}
