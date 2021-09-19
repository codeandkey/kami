#include "kami.h"

#include <iostream>
#include <torch/torch.h>

#include "log.h"
#include "model.h"

using namespace kami;
using namespace std;

int main(int argc, char** argv) {
    // Show program information
    
    cout << "> kami-grl " << VERSION << " build " << __DATE__ << endl;

    for (auto& author : AUTHORS) {
        cout << "> " << author << endl;
    }

    cout << "Starting." << endl;

    // Don't accept arguments
    if (argc > 1) {
        cout << "WARNING: Ignoring extra arguments" << endl;
    }

    // Show torch information
    cout << "Using torch version "
         << TORCH_VERSION_MAJOR << "."
         << TORCH_VERSION_MINOR << "."
         << TORCH_VERSION_PATCH << endl;

    // Show CUDA information
    if (torch::cuda::is_available()) {
        int devices = torch::cuda::device_count();
        bool cudnn = torch::cuda::cudnn_is_available();

        cout << "CUDA enabled, "
             << devices << " devices, "
             << "CUDNN " << (cudnn ? "enabled" : "disabled")
             << endl;
    } else {
        cout << "CUDA disabled" << endl;
    }

    return 0;
}
