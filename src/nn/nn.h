#pragma once

#include <string>
#include <vector>

#include <torch/nn.h>

using namespace torch;
using namespace torch::nn;

namespace kami {
    constexpr int FILTERS = 16;
    constexpr int RESIDUALS = 4;

    class NNResidual : public Module {
        private:
            Conv2d conv1{nullptr}, conv2{nullptr};
            ReLU relu{nullptr};
            BatchNorm2d batchnorm1{nullptr}, batchnorm2{nullptr};

        public:
            NNResidual();

            Tensor forward(Tensor x);
    };

    class NN : public Module {
        private:
            BatchNorm2d batchnorm{nullptr}, vbatchnorm{nullptr}, pbatchnorm{nullptr};
            Conv2d conv1{nullptr}, valueconv{nullptr}, policyconv{nullptr};
            Linear policyfc{nullptr}, valuefc1{nullptr}, valuefc2{nullptr};

            std::vector<ModuleHolder<NNResidual>> residuals;

            int width, height, features, psize;
            torch::Device device = torch::kCPU;

        public:
            NN(int width, int height, int features, int psize, bool force_cpu=false);

            std::vector<Tensor> forward(Tensor x);

            void infer(float* input, int batch, float* policy, float* value);

            bool isCUDA() { return device.is_cuda(); }

            void write(std::string path);
            void read(std::string path);

            void train(float* data, int batch);
    };
}
