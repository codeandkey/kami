#include "model.h"

kami::Residual::Residual(int filters, float dropout) {
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(filters, filters, 3)
            .padding_mode(torch::enumtype::kZeros())
            .padding(1)
            .stride(1)
    ));

    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(filters, filters, 3)
            .padding_mode(torch::enumtype::kZeros())
            .padding(1)
            .stride(1)
    ));

    bn1 = register_module("bn1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(filters)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(filters)));

    dropout_p = dropout;
}

torch::Tensor kami::Residual::forward(torch::Tensor x) {
    auto& skip = x;

    x = torch::relu(bn1->forward(conv1->forward(x)));
    x = torch::dropout(x, dropout_p, is_training());
    x = bn2->forward(conv2->forward(x));
    x = torch::add(skip, x);
    x = torch::relu(x);
    x = torch::dropout(x, dropout_p, is_training());

    return x;
}

kami::KamiNet::KamiNet(
    int num_residuals,
    int num_filters,
    float dropout,
    int policy,
    int features,
    int width,
    int height
) {
    // Initial convolution / bn
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(features, num_filters, 3)
            .padding_mode(torch::enumtype::kZeros())
            .padding(1)
            .stride(1)
    ));

    bn1 = register_module("bn1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters)));

    // Residuals

    for (int i = 0; i < num_residuals; ++i) {
        std::string rname = "residual_";
        rname += std::to_string(i);

        residuals.push_back(
            register_module(
                rname.c_str(),
                torch::nn::ModuleHolder<Residual>(num_filters, dropout)
            )
        );
    }

    // Value head

    vh_conv = register_module("vh_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(num_filters, 1, 1)
            .padding_mode(torch::enumtype::kZeros())
            .padding(1)
            .stride(1)
    ));

    vh_fc1 = register_module("vh_fc1", torch::nn::Linear(
        torch::nn::LinearOptions(features * width * height, 256)
    ));

    vh_fc2 = register_module("vh_fc2", torch::nn::Linear(
        torch::nn::LinearOptions(features * width * height, 1)
    ));

    // Policy head
    ph_conv = register_module("ph_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1, 2, 1)
    ));

    ph_fc = register_module("ph_fc", torch::nn::Linear(
        torch::nn::LinearOptions(2 * width * height, policy)
    ));

    ph_bn = register_module("ph_bn", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters)));

    dropout_p = dropout;
}

vector<torch::Tensor> kami::KamiNet::forward(torch::IValue inputs) {
    auto input_vec = inputs.toTensorList();

    auto x = input_vec.get(0);
    auto lmm = input_vec.get(1);

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = torch::dropout(x, dropout_p, is_training());
    
    for (auto& residual : residuals) {
        x = residual->forward(x);
    }

    vector<torch::Tensor> outputs;

    // Value head
    auto vh = x;
    vh = vh_conv->forward(vh);
    vh = torch::relu(vh);
    vh = torch::dropout(vh, dropout_p, is_training());
    vh = torch::flatten(vh);
    vh = vh_fc1->forward(vh);
    vh = torch::relu(vh);
    vh = torch::dropout(vh, dropout_p, is_training());
    vh = vh_fc2->forward(vh);
    vh = torch::tanh(vh);

    outputs.push_back(vh);

    // Policy head
    auto ph = x;
    ph = ph_conv->forward(ph);
    ph = ph_bn->forward(ph);
    ph = torch::relu(ph);
    ph = torch::dropout(ph, dropout_p, is_training());
    ph = torch::flatten(ph);
    ph = ph_fc->forward(ph);

    // Apply input mask

    return outputs;
}
