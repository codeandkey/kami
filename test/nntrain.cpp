#include "../src/nn/nn.h"
#include "../src/env.h"

#define TESTSIZE 512 // trajectories to send

using namespace kami;
using namespace std;

int main(int argc, char** argv) {
    float* inputs = new float[TESTSIZE * 8 * 8 * NFEATURES];
    float* policy = new float[TESTSIZE * PSIZE];
    float* value = new float[TESTSIZE];

    float* lmm = new float[TESTSIZE * PSIZE];

    for (int j = 0; j < TESTSIZE * PSIZE; ++j)
        lmm[j] = ((float) rand() / (float) RAND_MAX) > 0.95f ? 1.0f : 0.0f;

    for (int i = 0; i < TESTSIZE * 8 * 8 * NFEATURES; ++i)
        inputs[i] = (float) rand() / (float) RAND_MAX;

    for (int i = 0; i < TESTSIZE * PSIZE; ++i)
        policy[i] = (float) rand() / (float) RAND_MAX;

    for (int i = 0; i < TESTSIZE; ++i)
        value[i] = (float) rand() / (float) RAND_MAX;

    NN net(8, 8, NFEATURES, PSIZE);

    clock_t start = clock(), timer = start;
    net.train(TESTSIZE, inputs, lmm, policy, value);
    cout << "Finished in " << (double) (clock() - start) / (double) CLOCKS_PER_SEC << " seconds" << endl;

    delete[] inputs;
    delete[] policy;
    delete[] value;
    delete[] lmm;

    return 0;
}
