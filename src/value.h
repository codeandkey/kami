#pragma once

namespace kami {
template <typename P> class Value {
    public:
        float utility(P player) = 0;
}; // class Value
}; // namespace kami
