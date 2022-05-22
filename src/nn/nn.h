#include <string>
#include <iostream>
#include <vector>
#include <torch/nn.h>
#include <torch/cuda.h>

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
            NNResidual()
            {
                conv1 = register_module("conv1", Conv2d(Conv2dOptions(FILTERS, FILTERS, 3).padding(1).padding_mode(torch::kZeros)));
                conv2 = register_module("conv2", Conv2d(Conv2dOptions(FILTERS, FILTERS, 3).padding(1).padding_mode(torch::kZeros)));
                batchnorm1 = register_module("batchnorm1", BatchNorm2d(FILTERS));
                batchnorm2 = register_module("batchnorm2", BatchNorm2d(FILTERS));
                relu = register_module("relu", ReLU());
            }

            Tensor forward(Tensor x)
            {
                auto skip = x;

                x = relu(batchnorm1(conv1(x)));
                x = skip + batchnorm2(conv2(x));
                
                return x;
            }
    };

    class NN : public Module {
        private:
            BatchNorm2d batchnorm{nullptr}, vbatchnorm{nullptr}, pbatchnorm{nullptr};
            Conv2d conv1{nullptr}, valueconv{nullptr}, policyconv{nullptr};
            Linear policyfc{nullptr}, valuefc1{nullptr}, valuefc2{nullptr};

            std::vector<ModuleHolder<NNResidual>> residuals;

            int width, height, features, psize;
            torch::Device device;
        public:
            NN(int width, int height, int features, int psize, torch::Device device) :
                width(width),
                height(height),
                features(features),
                psize(psize),
                device(device)
            {
                batchnorm = register_module("batchnorm1", BatchNorm2d(FILTERS));
                vbatchnorm = register_module("vbatchnorm", BatchNorm2d(3));
                pbatchnorm = register_module("pbatchnorm", BatchNorm2d(32));
                conv1 = register_module("conv1", Conv2d(Conv2dOptions(features, FILTERS, 3).padding(1).padding_mode(torch::kZeros)));
                valueconv = register_module("valueconv", Conv2d(Conv2dOptions(FILTERS, 3, 1)));
                policyconv = register_module("policyconv", Conv2d(Conv2dOptions(FILTERS, 32, 1)));
                policyfc = register_module("policyfc", Linear(32 * width * height, psize));
                valuefc1 = register_module("valuefc1", Linear(3 * width * height, 128));
                valuefc2 = register_module("valuefc2", Linear(128, 1));

                // residual layers
                for (int i = 0; i < RESIDUALS; ++i)
                    residuals.push_back(register_module("residual" + std::to_string(i), ModuleHolder<NNResidual>()));

                to(device);
            }

            std::vector<Tensor> forward(Tensor x)
            {
                // initial convolution
                x = x.permute({0, 3, 1, 2});
                x = conv1->forward(x);
                x = batchnorm->forward(x);
                x = torch::relu(x);

                // apply residuals
                for (int i = 0; i < RESIDUALS; ++i)
                    x = residuals[i]->forward(x);

                // policy head
                Tensor ph = policyconv->forward(x);
                ph = pbatchnorm->forward(ph);
                ph = torch::relu(ph);
                ph = ph.flatten(1);
                ph = policyfc->forward(ph);

                // value head
                Tensor vh = valueconv->forward(x);
                vh = vbatchnorm->forward(vh);
                vh = torch::relu(vh);
                vh = vh.flatten(1);
                vh = valuefc1->forward(vh);
                vh = valuefc2->forward(vh);
                vh = vh.tanh();

                return { ph, vh };
            }

            void infer(float* input, int batch, float* policy, float* value)
            {
                Tensor inputs = torch::from_blob(input, { batch, width, height, features }, {1, 1, 1, 1}, torch::kCPU);
                inputs.reshape({ batch, width, height, features });

                inputs = inputs.to(device, torch::kFloat32);

                std::vector<Tensor> outputs = forward(inputs);

                float* policy_data = outputs[0].cpu().data_ptr<float>();
                float* value_data = outputs[1].cpu().data_ptr<float>();

                memcpy(policy, policy_data, batch * psize * sizeof(float));
                memcpy(value, value_data, batch * sizeof(float));
            }
    };
}
