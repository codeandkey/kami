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

    class NNModule : public Module {
        private:
            BatchNorm2d batchnorm{nullptr}, vbatchnorm{nullptr}, pbatchnorm{nullptr};
            Conv2d conv1{nullptr}, valueconv{nullptr}, policyconv{nullptr};
            Linear policyfc{nullptr}, valuefc1{nullptr}, valuefc2{nullptr};
            std::vector<ModuleHolder<NNResidual>> residuals;

            int width, height, features, psize;

        public:
            NNModule(int width, int height, int features, int psize);

            std::vector<torch::Tensor> forward(std::vector<torch::IValue> x);
            Tensor loss(Tensor& p, Tensor& v, Tensor& obsp, Tensor& obsv, Tensor& lmm);
    };

    class NN {
        private:
            std::shared_ptr<NNModule> mod;
            int width, height, features, psize;

            std::shared_mutex mut;
            int generation;

            torch::Device device = torch::kCPU;
        public:
            NN(int width, int height, int features, int psize, bool force_cpu=false);
            NN(NN* other);

            int get_generation() { 
                mut.lock_shared();
                int ret = generation;
                mut.unlock_shared();

                return ret; 
            }

            torch::Device get_device() { return device; }
            bool isCUDA() { return device.is_cuda(); }
            int obsize() const { return width * height * features; }
            int polsize() const { return psize; }

            void infer(float* input, float* inplmm, int batch, float* policy, float* value);
            void train(int trajectories, float* inputs, float* lmm, float* obs_p, float* obs_v);

            void read(std::string path);
            void write(std::string path);

            NN* clone();
    };
}
