#pragma once

#include "env.h"
#include "options.h"

#include <cmath>
#include <random>
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>

namespace kami {
struct Node {
    int n = 0;
    float w = 0.0f;
    float p = 0.0f;
    int action = -1;
    std::vector<Node*> children;
    Node* parent = nullptr;

    float turn;
    float q(float def = 1.0f) { return n > 0 ? w / n : def; }

    void clean()
    {
        for (auto& c : children)
        {
            c->clean();
            delete c;
        }
    }

    void backprop(float value)
    {
        n += 1;
        w += 0.5f + (value * turn) / 2.0f;

        if (parent)
            parent->backprop(value);
    }

    std::string debug(Env* e)
    {
        std::stringstream out;
        float value;

        out << std::setw(6) << e->debug_action(action);
        out << " Visits: " << std::setw(4) << std::to_string(n);
        out << " Average: " << std::to_string(q());
        out << " Policy: " << std::to_string(p);
        out << " Turn: " << std::to_string(turn);

        e->push(action);

        if (e->terminal(&value))
            out << " Terminal: " << std::to_string(value);

        e->pop();

        return out.str();
    }
};

class MCTS {
    private:
        Env env;
        Node* target = nullptr;
        double cPUCT;
        bool force_expand_unvisited;
        float unvisited_node_value;
        float bootstrap_weight;

        float bootstrap_window;
        float bootstrap_amp;
        float noise_weight;
        float noise_alpha;
        int scale_cpuct_by_actions;

        std::mt19937 rng;

    public:
        Node* root = nullptr;
        MCTS()
        {
            root = new Node();
            root->turn = -env.turn();
            cPUCT = options::getFloat("cpuct", 1.0f);
            force_expand_unvisited = options::getInt("force_expand_unvisited", 0);
            unvisited_node_value = (float) options::getInt("unvisited_node_value_pct", 100) / 100.0f;
            bootstrap_weight = (float) options::getInt("bootstrap_weight", 0) / 100.0f;
            bootstrap_window = (float) options::getInt("bootstrap_window", 1600);
            bootstrap_amp = (float) options::getInt("bootstrap_amp_pct", 75) / 100.0f;
            scale_cpuct_by_actions = options::getInt("scale_cpuct_by_actions", 0);
            noise_alpha = options::getFloat("mcts_noise_alpha", 0.05f);
            noise_weight = options::getFloat("mcts_noise_weight", 0.05f);

            rng.seed(time(NULL));
        }

        ~MCTS()
        {
            if (root)
            {
                root->clean();
                delete root;
            }
        }

        int n() { return root->n; }

        void push(int action)
        {
            Node* next = nullptr;

            for (auto& c : root->children)
            {
                if (c->action == action)
                    next = c;
                else
                {
                    c->clean();
                    delete c;
                }
            }

            if (!next)
                throw std::runtime_error("no child for action");

            delete root;
            root = next;
            root->parent = nullptr;
            env.push(action);
        }

        int pick(float alpha = 0.0f) {
            if (!root->children.size())
                throw std::runtime_error("no children to pick from");

            if (alpha < 0.1f)
            {
                int best_n = 0;
                int best_action = -1;

                for (auto& c : root->children)
                {
                    if (c->n > best_n)
                    {
                        best_n = c->n;
                        best_action = c->action;
                    }
                }

                return best_action;
            }

            double dist[root->children.size()], length = 0.0f;

            for (unsigned i = 0; i < root->children.size(); ++i)
            {
                Node* child = root->children[i];

                double d = pow(child->n, 1.0f / alpha);

                dist[i] = d;
                length += d;
            }

            for (auto& d : dist)
                d /= length;

            double ind = (double) rand() / (double) RAND_MAX;

            for (unsigned i = 0; i < root->children.size(); ++i)
            {
                ind -= dist[i];

                if (ind <= 0.0)
                    return root->children[i]->action;
            }

            return root->children.back()->action;
        }

        bool select(float* obs)
        {
            if (!target)
                target = root;

            // If no children, need to expand
            if (target->children.empty())
            {
                // Test terminal state
                float value;
                if (env.terminal(&value))
                {
                    target->backprop(value);

                    while (target != root)
                    {
                        env.pop();
                        target = target->parent;
                    }

                    target = nullptr;
                    return false;
                }

                env.observe(obs);
                return true;
            }

            // Iterate children
            double best_uct = -1000.0;
            Node* best_child = nullptr;

            float cpuct = cPUCT;

            if (scale_cpuct_by_actions)
                cpuct /= (float) target->children.size();

            for (auto& c : target->children)
            {
                // Force expanding unvisisted children
                if (force_expand_unvisited && !c->n)
                {
                    target = c;
                    env.push(c->action);
                    return select(obs);
                }

                double uct = c->q(unvisited_node_value * c->turn) + c->p * cpuct * sqrt(target->n) / (double) (c->n + 1);

                if (uct > best_uct)
                {
                    best_child = c;
                    best_uct = uct;
                }
            }

            #ifndef NDEBUG
            if (!best_child)
            {
                for (auto& c : target->children)
                    std::cerr << "child " << c->action << " : q=" << c->q(unvisited_node_value) << ", p=" << c->p << ", pmul=" << cpuct * sqrt(target->n) / (double) (c->n + 1) << std::endl;

                throw std::runtime_error("no best child to select, but children present!");
            }
            #endif

            env.push(best_child->action);
            target = best_child;
            return select(obs);
        }

        void expand(float* policy, float value, bool disable_bootstrap=false)
        {
            std::vector<int> actions = env.actions();

            #ifndef NDEBUG
            if (!actions.size())
                throw std::runtime_error("expand() called with no actions");

            float tsum = 0.0f;
            for (int i = 0; i < PSIZE; ++i)
                tsum += policy[i];

            if (tsum <= 0.999f)
                throw std::runtime_error("softmax sums to " + std::to_string(tsum));
            #endif

            float ptotal = 0.0f;

            for (int action : actions)
                ptotal += policy[action];

            // Generate noise for each action
            std::vector<float> noise(actions.size(), 0.0f);
            float total_noise = 0.0f;

            for (int i = 0; i < noise.size(); ++i)
            {
                std::gamma_distribution<> dist(1.0f, 1.0f);
                noise[i] = dist(rng);
                total_noise += noise[i];
            }

            for (int i = 0; i < actions.size(); ++i)
            {
                Node* new_child = new Node();

                new_child->action = actions[i];
                new_child->parent = target;
                new_child->turn = -target->turn;
                new_child->p = (1 - noise_weight) * policy[actions[i]] / ptotal + noise_weight * (noise[i] / total_noise);

                #ifndef NDEBUG
                    if (policy[actions[i]] < 0.0f)
                        throw std::runtime_error("negative policy detected: " + std::to_string(policy[actions[i]]));

                    if (std::isnan(policy[actions[i]]))
                        throw std::runtime_error("NaN policy detected");
                #endif

                target->children.push_back(new_child);
            }

            // The NN outputs a value relative to this action. We are looking
            // for the absolute value of the position. Then we simply normalize
            // the NN output and then apply the unflipped neocortex evaluation.

            value *= target->turn;

            if (!disable_bootstrap && bootstrap_weight > 0.0f)
                value = (1 - bootstrap_weight) * value + bootstrap_weight * env.bootstrap_value(bootstrap_window) * bootstrap_amp;

            target->backprop(value);

            while (target != root)
            {
                env.pop();
                target = target->parent;
            }

            target = nullptr;
        }

        Env& get_env() { return env; }

        void reset() {
            env = Env();
            target = nullptr;
            root->clean();

            delete root;
            root = new Node();
            root->turn = -env.turn();
        }

        void snapshot(float* pspace)
        {
            for (int i = 0; i < PSIZE; ++i)
                pspace[i] = 0.0f;

            for (auto& c : root->children)
                pspace[c->action] = (float) c->n / (float) (root->n - 1);
        }
};
}
