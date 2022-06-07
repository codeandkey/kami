#pragma once

#include "nn/nn.h"

namespace kami {
    constexpr int EVAL_RUNS  = 10;
    constexpr int EVAL_BATCH = 16;
    constexpr int EVAL_NODES = 512;
    constexpr float EVAL_CRIT = 0.54f;

    bool eval(NN* current_model, NN* candidate_model);
}
