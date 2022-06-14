#include "nn.h"
#include "../options.h"

#include <memory>
#include <random>
#include <torch/cuda.h>

#include <torch/nn/modules/loss.h>
#include <torch/optim.h>
#include <torch/optim/sgd.h>
#include <torch/serialize.h>

using namespace kami;
using namespace torch;
using namespace torch::nn;
using namespace std;

NNResidual::NNResidual(int filters)
{
    conv1 = register_module("conv1", Conv2d(Conv2dOptions(filters, filters, 3).padding(1).padding_mode(torch::kZeros)));
    conv2 = register_module("conv2", Conv2d(Conv2dOptions(filters, filters, 3).padding(1).padding_mode(torch::kZeros)));
    batchnorm1 = register_module("batchnorm1", BatchNorm2d(filters));
    batchnorm2 = register_module("batchnorm2", BatchNorm2d(filters));
}

Tensor NNResidual::forward(Tensor x)
{
    auto skip = x;

    x = torch::relu(batchnorm1(conv1(x)));
    x = skip + batchnorm2(conv2(x));
    
    return x;
}

NNModule::NNModule(int width, int height, int features, int psize) :
    width(width),
    height(height),
    features(features),
    psize(psize)
{
    int filters = options::getInt("filters", 16);
    int nresiduals = options::getInt("residuals", 4);

    batchnorm = register_module("batchnorm1", BatchNorm2d(filters));
    vbatchnorm = register_module("vbatchnorm", BatchNorm2d(3));
    pbatchnorm = register_module("pbatchnorm", BatchNorm2d(32));
    conv1 = register_module("conv1", Conv2d(Conv2dOptions(features, filters, 3).padding(1).padding_mode(torch::kZeros)));
    valueconv = register_module("valueconv", Conv2d(Conv2dOptions(filters, 3, 1)));
    policyconv = register_module("policyconv", Conv2d(Conv2dOptions(filters, 32, 1)));
    policyfc = register_module("policyfc", Linear(32 * width * height, psize));
    valuefc1 = register_module("valuefc1", Linear(3 * width * height, 128));
    valuefc2 = register_module("valuefc2", Linear(128, 1));

    // residual layers
    for (int i = 0; i < nresiduals; ++i)
        residuals.push_back(register_module("residual" + to_string(i), ModuleHolder<NNResidual>(filters)));
}

vector<Tensor> NNModule::forward(Tensor x)
{
    // initial convolution
    x = x.permute({0, 3, 1, 2});
    x = conv1->forward(x);
    x = batchnorm->forward(x);
    x = torch::relu(x);

    // apply residuals
    for (int i = 0; i < residuals.size(); ++i)
        x = residuals[i]->forward(x);

    // policy head
    Tensor ph = policyconv->forward(x);
    ph = pbatchnorm->forward(ph);
    ph = torch::relu(ph);
    ph = ph.flatten(1);
    ph = policyfc->forward(ph);
    ph = torch::softmax(ph, 1);

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

Tensor NNModule::loss(Tensor& p, Tensor& v, Tensor& obsp, Tensor& obsv)
{
    // Value loss: MSE
    Tensor value_loss = mse_loss(v, obsv).sum();

    // Policy loss -(obsp . log(p)) [maximize directional similarity to p-obsp]
    Tensor policy_loss = obsp
        .mul(torch::log(p + 0.0001))
        .sum()
        .neg();

    return policy_loss.add(value_loss);
}

NN::NN(int width, int height, int features, int psize, bool force_cpu) :
    width(width),
    height(height),
    features(features),
    psize(psize)
{
    mod = make_shared<NNModule>(width, height, features, psize);
    generation = 0;

    // try cuda
    if (torch::cuda::is_available() && !force_cpu)
    {
        device = torch::Device(kCUDA, 0);
        mod->to(device);
        return;
    }

    if (!force_cpu)
        cerr << "WARNING: CUDA not available, NN operations will be slow\n";
}

NN::NN(NN* other)
{
    other->mut.lock_shared();

    width = other->width;
    height = other->height;
    features = other->features;
    psize = other->psize;
    device = other->device;
    generation = other->generation;

    mod = make_shared<NNModule>(width, height, features, psize);

    std::stringstream sstr;

    // Just serialize and de-serialize the model -- Cloneable<> is a mess
    torch::save(other->mod, sstr);
    torch::load(mod, sstr);

    other->mut.unlock_shared();
    
    mod->to(device);
}

void NN::infer(float* input, float* inplmm, int batch, float* policy, float* value)
{
    Tensor inputs = torch::from_blob(input, { batch, width, height, features }, torch::kCPU);
    Tensor lmm = torch::from_blob(inplmm, { batch, psize }, torch::kCPU);
    inputs = inputs.reshape({ batch, width, height, features });

    inputs = inputs.to(device, torch::kFloat32);
    lmm = lmm.to(device, torch::kFloat32);

    vector<Tensor> outputs;

    {
        torch::NoGradGuard guard;
        mut.lock_shared();
        outputs = mod->forward(inputs);
        mut.unlock_shared();
    }

    Tensor ph = outputs[0], vh = outputs[1];

    ph = ph.mul(lmm);

    Tensor psum = at::sum(ph, 1, true).clamp(0.0001).expand({-1, psize});

    #ifndef NDEBUG
        if ((psum == 0).any().item().toBool())
            throw runtime_error("infer: policy sum contains a zero");
    #endif

    ph = ph.div(psum);

    ph = ph.cpu();
    vh = vh.cpu();
    
    #ifndef NDEBUG
        if (ph.isnan().any().item().toBool())
            throw runtime_error("infer: final policy output contains NaN");

        if (vh.isnan().any().item().toBool())
            throw runtime_error("infer: final value output contains NaN");
    #endif

    float* policy_data = ph.data_ptr<float>();
    float* value_data = vh.data_ptr<float>();

    memcpy(policy, policy_data, batch * psize * sizeof(float));
    memcpy(value, value_data, batch * sizeof(float));
}

void NN::write(string path)
{
    mut.lock_shared();

    serialize::OutputArchive a;
    mod->save(a);

    a.write("generation", IValue(generation));

    a.save_to(path);
    mut.unlock_shared();
}

void NN::read(string path)
{
    mut.lock();
    try {
        serialize::InputArchive i;
        i.load_from(path);

        IValue genvalue;
        i.read("generation", genvalue);

        generation = genvalue.toInt();

        mod->load(i);
        mut.unlock();
    } catch (exception& e) {
        mut.unlock();
        throw e;
    }
}

void NN::train(int trajectories, float* inputs, float* obs_p, float* obs_v, bool detect_anomaly)
{
    mut.lock();

    // Detect anomalies
    if (detect_anomaly)
        torch::autograd::AnomalyMode::set_enabled(true);

    float lr = (float) options::getInt("training_mlr", 5) / 1000.0f;
    int epochs = options::getInt("training_epochs", 8);
    int tbatch = options::getInt("training_batchsize", 8);

    // initialize optimizer
    optim::SGD optimizer(
        mod->parameters(), optim::SGDOptions(lr)
    );

    // magic batch picker
    vector<int> picker(trajectories, 0);

    auto rng = default_random_engine {};

    for (int i = 0; i < (int) picker.size(); ++i)
        picker[i] = i;

    int trainbatch_start = 0;

    float firstloss, lastloss;

    // start epochs
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // prepare picker
        shuffle(picker.begin(), picker.end(), rng);

        // training batch data
        float next_input[tbatch][width][height][features];
        float next_policy[tbatch][psize];
        float next_value[tbatch];

        vector<Tensor> training_inputs;
        vector<Tensor> training_obsp;
        vector<Tensor> training_obsv;

        int batch_base = 0;

        while (batch_base <= picker.size() - 1)
        {
            // build batch
            int i;

            // copy observations
            for (i = 0; i < tbatch && i + batch_base <= picker.size() - 1; ++i)
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

            batch_base += i;

            // build tensors
            training_inputs.push_back(torch::from_blob(
                next_input, 
                {tbatch, width, height, features},
                kCPU
            ).to(device, kFloat32));

            training_obsp.push_back(torch::from_blob(
                next_policy, 
                {tbatch, psize},
                kCPU
            ).to(device, kFloat32));

            training_obsv.push_back(torch::from_blob(
                next_value, 
                {tbatch, 1},
                kCPU
            ).to(device, kFloat32));
        }

        // train
        float avgloss = 0.0f;

        float epfirstloss;
        float eplastloss;

        for (int i = 0; i < training_inputs.size(); ++i)
        {
            mod->zero_grad();

            if (detect_anomaly)
            {
                if (training_inputs[i].isnan().any().cpu().item().toBool())
                    throw runtime_error("training input ind " + to_string(i) + " contains NaN");
            }

            vector<Tensor> outputs = mod->forward(training_inputs[i]);

            if (detect_anomaly)
            {
                if (outputs[1].isnan().any().cpu().item().toBool())
                    throw runtime_error("forward value output contains NaN");

                if (outputs[0].isnan().any().cpu().item().toBool())
                    throw runtime_error("forward policy output contains NaN");
            }

            Tensor lossval = mod->loss(
                outputs[0],
                outputs[1],
                training_obsp[i],
                training_obsv[i]
            );

            lossval.backward();
            optimizer.step();

            float thisloss = lossval.cpu().item<float>();
            avgloss += thisloss;

            if (!i) epfirstloss = thisloss;
            if (i == training_inputs.size() - 1) eplastloss = thisloss;
        }

        avgloss /= (float) training_inputs.size();
        cout << "Epoch " << epoch + 1 << "/" << epochs << ": loss " << epfirstloss << " => " << eplastloss << ", " << training_inputs.size() << " batches" << endl;

        if (!epoch)
            firstloss = avgloss;

        lastloss = avgloss;
    }

    ++generation;
    cout << "Generated model " << generation << ", average loss " << firstloss << " to " << lastloss << " over " << epochs << " epochs\n";

    mut.unlock();
}
