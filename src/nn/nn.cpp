#include "nn.h"

#include <random>
#include <torch/cuda.h>
#include <torch/nn/modules/loss.h>
#include <torch/optim.h>
#include <torch/optim/sgd.h>
#include <torch/serialize/output-archive.h>

using namespace kami;
using namespace torch;
using namespace torch::nn;

NNResidual::NNResidual()
{
    conv1 = register_module("conv1", Conv2d(Conv2dOptions(FILTERS, FILTERS, 3).padding(1).padding_mode(torch::kZeros)));
    conv2 = register_module("conv2", Conv2d(Conv2dOptions(FILTERS, FILTERS, 3).padding(1).padding_mode(torch::kZeros)));
    batchnorm1 = register_module("batchnorm1", BatchNorm2d(FILTERS));
    batchnorm2 = register_module("batchnorm2", BatchNorm2d(FILTERS));
    relu = register_module("relu", ReLU());
}

Tensor NNResidual::forward(Tensor x)
{
    auto skip = x;

    x = relu(batchnorm1(conv1(x)));
    x = skip + batchnorm2(conv2(x));
    
    return x;
}

NN::NN(int width, int height, int features, int psize, bool force_cpu) :
    width(width),
    height(height),
    features(features),
    psize(psize)
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

    // try cuda
    if (torch::cuda::is_available() && !force_cpu)
    {
        device = torch::Device(kCUDA, 0);
        to(device);
        return;
    }

    if (!force_cpu)
        std::cerr << "WARNING: CUDA not available, NN operations will be slow\n";
}

std::vector<Tensor> NN::forward(std::vector<IValue> inputs)
{
    Tensor x = inputs[0].toTensor();
    Tensor lmm = inputs[1].toTensor();

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
    ph = torch::relu(ph).exp().mul(lmm);

    Tensor total = at::sum(ph, 1, true);

    ph = ph.div(total.expand({-1, psize}));

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

void NN::infer(float* input, float* inplmm, int batch, float* policy, float* value)
{
    Tensor inputs = torch::from_blob(input, { batch, width, height, features }, {1, 1, 1, 1}, torch::kCPU);
    Tensor lmm = torch::from_blob(inplmm, { batch, psize }, {1, 1}, torch::kCPU);
    inputs = inputs.reshape({ batch, width, height, features });
    lmm = lmm.reshape({ batch, psize });

    inputs = inputs.to(device, torch::kFloat32);
    lmm = lmm.to(device, torch::kFloat32);

    std::vector<Tensor> outputs = forward({ inputs, lmm });

    outputs[0] = outputs[0].cpu();
    outputs[1] = outputs[1].cpu();

    float* policy_data = outputs[0].data_ptr<float>();
    float* value_data = outputs[1].data_ptr<float>();

    memcpy(policy, policy_data, batch * psize * sizeof(float));
    memcpy(value, value_data, batch * sizeof(float));
}

void NN::write(std::string path)
{
    serialize::OutputArchive a;
    save(a);

    a.save_to(path);
}

void NN::read(std::string path)
{
    serialize::InputArchive i;
    i.load_from(path);

    load(i);
}

void NN::train(int trajectories, float* inputs, float* lmm, float* obs_p, float* obs_v, int epochs)
{
    // initialize optimizer
    optim::SGD optimizer(
        parameters(), optim::SGDOptions(LEARNING_RATE)
    );

    // magic batch picker
    std::vector<int> picker(trajectories, 0);

    auto rng = std::default_random_engine {};

    for (int i = 0; i < (int) picker.size(); ++i)
        picker[i] = i;

    int trainbatch_start = 0;

    float firstloss, lastloss;

    // start epochs
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // prepare picker
        std::shuffle(picker.begin(), picker.end(), rng);

        // training batch data
        float next_input[TRAIN_BATCHSIZE][width][height][features];
        float next_lmm[TRAIN_BATCHSIZE][psize];
        float next_policy[TRAIN_BATCHSIZE][psize];
        float next_value[TRAIN_BATCHSIZE];

        std::vector<Tensor> training_inputs;
        std::vector<Tensor> training_lmm;
        std::vector<Tensor> training_obsp;
        std::vector<Tensor> training_obsv;

        int batch_base = 0;

        while (batch_base <= picker.size() - 1)
        {
            // build batch
            int i;

            for (i = 0; i < TRAIN_BATCHSIZE && i + batch_base <= picker.size() - 1; ++i)
                memcpy(
                    &next_input[i][0][0],
                    inputs + picker[batch_base + i] * width * height * features,
                    sizeof(float) * width * height * features 
                );

            // copy policies
            for (int j = 0; j < i; ++j)
                memcpy(
                    &next_policy[j][0],
                    obs_p + picker[batch_base + j] * psize,
                    sizeof(float) * psize
                );

            // copy values 
            for (int j = 0; j < i; ++j)
                next_value[j] = obs_v[picker[batch_base + j]];

            // copy lmm 
            for (int j = 0; j < i; ++j)
                memcpy(
                    &next_lmm[j][0],
                    lmm + picker[batch_base + j] * psize,
                    sizeof(float) * psize
                );

            batch_base += i;

            // build tensors
            training_inputs.push_back(torch::from_blob(
                next_input, 
                {TRAIN_BATCHSIZE, width, height, features},
                kCPU
            ).to(device, kFloat32));

            training_obsp.push_back(torch::from_blob(
                next_policy, 
                {TRAIN_BATCHSIZE, psize},
                kCPU
            ).to(device, kFloat32));

            training_obsv.push_back(torch::from_blob(
                next_value, 
                {TRAIN_BATCHSIZE, 1},
                kCPU
            ).to(device, kFloat32));

            training_lmm.push_back(torch::from_blob(
                next_lmm, 
                {TRAIN_BATCHSIZE, psize},
                kCPU
            ).to(device, kFloat32));
        }

        // train
        float avgloss;

        for (int i = 0; i < training_inputs.size(); ++i)
        {
            zero_grad();

            std::vector<IValue> x = { training_inputs[i], training_lmm[i] };
            std::vector<Tensor> outputs = forward(x);

            Tensor lossval = loss(
                outputs[0],
                outputs[1],
                training_obsp[i],
                training_obsv[i],
                training_lmm[i]
            );

            lossval.backward();
            optimizer.step();

            float thisloss = lossval.cpu().item<float>();
            avgloss += thisloss;
        }

        avgloss /= (float) training_inputs.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << ": loss " << avgloss << std::endl;

        if (!epoch)
            firstloss = avgloss;

        lastloss = avgloss;
    }

    std::cout << "Finished training, average loss " << firstloss << " to " << lastloss << " over " << epochs << " epochs" << std::endl;
}

Tensor NN::loss(Tensor& p, Tensor& v, Tensor& obsp, Tensor& obsv, Tensor& lmm)
{
    // Value loss: MSE
    Tensor value_loss = mse_loss(v, obsv).sum();

    // Policy loss -(obsp . log(p)) [maximize directional similarity to p-obsp]
    Tensor policy_loss = obsp
        .mul(lmm)
        .mul(torch::log(p + 0.0001))
        .sum()
        .neg();

    return policy_loss.add(value_loss);
}
