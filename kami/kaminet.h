#pragma once

#include <torch/nn.h>

/**
 * Kami generalized Torch model type
 */

namespace kami {
    struct Residual : torch::nn::Module {
        Residual(int filters, float dropout);
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
        float dropout_p;
    };

    struct KamiNet : torch::nn::Module {
        KamiNet(
            int num_residuals,
            int num_filters,
            float dropout,
            int policy,
            int features,
            int width,
            int height
        );

        vector<torch::Tensor> forward(torch::IValue inputs);
        std::vector<torch::nn::ModuleHolder<Residual>> residuals;

        torch::nn::Conv2d conv1{nullptr}, vh_conv{nullptr}, ph_conv{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, vh_bn{nullptr}, ph_bn{nullptr};
        torch::nn::Linear vh_fc1{nullptr}, vh_fc2{nullptr}, ph_fc{nullptr};

        float dropout_p;
    };
}