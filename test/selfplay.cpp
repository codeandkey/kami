#include "../src/selfplay.h"
#include "../src/env.h"

#include <chrono>

using namespace kami;
using namespace std;

int main(int argc, char** argv) {
    NN model(8, 8, NFEATURES, PSIZE);
    Selfplay S(&model, 16, 50);
   
    S.start(1);

    // Don't busywait

    while (1)
        this_thread::sleep_for(chrono::seconds(15));

    S.stop();

    return 0;
}
