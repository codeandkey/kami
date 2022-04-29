#pragma once

namespace kami {
class MCTSNode {
    private:
        int incoming_player;
        int visits;
        float value, policy;

        MCTSNode* first_child, *parent, *sibling;

    public:
        MCTSNode(int incoming_color);

        void select(Environment* state);
        void rollout(Value* value, Policy* policy);

        MCTSNode* getBestChild();
        int getVisits() { return visits; }
        float getValueAvg() { return visits ? value / visits : 0; }

    protected:
        MCTSNode* getFirstChild() { return first_child; }
        MCTSNode* getParent() { return parent; }
        MCTSNode* getSibling() { return sibling; }
        float getPolicy() { return policy; }

}; // class MCTSNode
}; // namespace kami
