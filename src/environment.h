#pragma once

#include <string>
#include <vector>

namespace kami {
template <typename V> class Environment {
    public:
        static std::string name() = 0;
        static int psize() = 0;
        std::vector<A> actions() = 0;
        void mask(int* dst);
        bool terminal(V* value) = 0;
        void push(A action);
        void pop();
        float* observe();
        std::vector<int> shape();
}; // class Environment
}; // namespace kami
