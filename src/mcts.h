#pragma once

#include "env.h"

#include <cmath>
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
    float q() { return n > 0 ? w / n : 0; }

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

        out << std::setw(6) << e->decode(action).NaturalOut(&e->board);
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
        Env* env = nullptr;
        Node* target = nullptr;

    public:
        Node* root = nullptr;
        MCTS(Env* initial)
        {
            env = initial;

            root = new Node();
            root->turn = -env->turn();
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
            env->push(action);
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

                double d = ((double) child->n / (double) root->n) * pow(child->n, alpha);

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
            static constexpr double cPUCT = 1.0;

            if (!target)
                target = root;

            // If no children, need to expand
            if (target->children.empty())
            {
                // Test terminal state
                float value;
                if (env->terminal(&value))
                {
                    target->backprop(value);

                    while (target != root)
                    {
                        env->pop();
                        target = target->parent;
                    }

                    target = nullptr;
                    return false;
                }

                env->observe(obs);
                return true;
            }

            // Iterate children
            double best_uct = -1000.0;
            Node* best_child = nullptr;

            for (auto& c : target->children)
            {
                if (!c->n)
                {
                    target = c;
                    env->push(c->action);
                    return select(obs);
                }

                double uct = c->q() + c->p * cPUCT * sqrt(target->n) / (double) (c->n + 1);

                if (uct > best_uct)
                {
                    best_child = c;
                    best_uct = uct;
                }
            }

            env->push(best_child->action);
            target = best_child;
            return select(obs);
        }

        void expand(float* policy, float value)
        {
            for (auto& action : env->actions())
            {
                Node* new_child = new Node();

                new_child->action = action;
                new_child->parent = target;
                new_child->turn = -target->turn;
                new_child->p = policy[action];

                target->children.push_back(new_child);
            }

            target->backprop(value);

            while (target != root)
            {
                env->pop();
                target = target->parent;
            }

            target = nullptr;
        }
};
}