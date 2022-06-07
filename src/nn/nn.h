#pragma once

#include <atomic>
#include <string>
#include <vector>
#include <shared_mutex>

#include <torch/nn.h>

using namespace torch;
using namespace torch::nn;

namespace kami {
    class NNResidual : public Module {
        private:
            Conv2d conv1{nullptr}, conv2{nullptr};
            ReLU relu{nullptr};
            BatchNorm2d batchnorm1{nullptr}, batchnorm2{nullptr};

        public:
            NNResidual(int filters);

            Tensor forward(Tensor inputs);
    };

    class NN : public Module {
        private:
            BatchNorm2d batchnorm{nullptr}, vbatchnorm{nullptr}, pbatchnorm{nullptr};
            Conv2d conv1{nullptr}, valueconv{nullptr}, policyconv{nullptr};
            Linear policyfc{nullptr}, valuefc1{nullptr}, valuefc2{nullptr};

            std::vector<ModuleHolder<NNResidual>> residuals;

            int width, height, features, psize;
            torch::Device device = torch::kCPU;

            std::shared_mutex mut;

            Tensor loss(Tensor& p, Tensor& v, Tensor& obsp, Tensor& obsv, Tensor& lmm);

        public:
            NN(int width, int height, int features, int psize, bool force_cpu=false, int generation = 0);

            std::vector<torch::Tensor> forward(std::vector<torch::IValue> x);

            void infer(float* input, float* inplmm, int batch, float* policy, float* value);

            bool isCUDA() { return device.is_cuda(); }

            void write(std::string path);
            void read(std::string path);

            int obsize() const { return width * height * features; }
            int polsize() const { return psize; }

            std::atomic<int> generation;

            void train(int trajectories, float* inputs, float* lmm, float* obs_p, float* obs_v);
    };
}
