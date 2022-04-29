#pragma once

namespace kami {
template <typename A> class Policy {
    public:
        float policy_for_action(A action) = 0;
}; // class Policy 
}; // namespace kami
