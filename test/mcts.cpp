#include <algorithm>
#include <iostream>
#include <iomanip>

#include <cmath>
#include <ctime>

#include "../src/mcts.h"
#include "../src/env.h"

using namespace kami;
using namespace std;

int main()
{
    Env env;
    MCTS tree(&env);
    float value;
    std::string desc;

    srand(time(NULL));

    while (!env.terminal_str(&value, desc))
    {
        cout << "==============================\n";
        cout << env.print() << "\n";
        cout << "legal actions: ";

        for (int i : env.actions()) 
        {
            if (env.decode(i) != env.decode(env.encode(env.decode(i))))
            {
                cout << i << " failed, decodes to " << env.decode(i).TerseOut() << "\n";
                cout << ", but " << env.decode(i).TerseOut() << " encodes to " << env.encode(env.decode(i)) << "\n";
            }

            cout << " " << env.decode(i).NaturalOut(&env.board);
        }

        cout << "\n";

        while (tree.n() < 1024)
        {
            float observation[8 * 8 * NFEATURES];
            float lmm[PSIZE];

            while (!tree.select(observation, lmm) && tree.n() < 1024);
            if (tree.n() >= 1024) break;

/*
            cout << "observing square A1:";

            for (int i = 0; i < NFEATURES; ++i)
                cout << " " << (int) observation[i];

            cout << "";
*/
            float policy[4096 + 4 * 22];
            float p = 1.0f / (float) env.actions().size();

            for (int i = 0; i < 4096 + 4 * 22; ++i)
                policy[i] = p;

            double value = (((double) rand() / (double) RAND_MAX) * 2.0 - 1.0);

            tree.expand(policy, value);
        }

        std::sort(tree.root->children.begin(), tree.root->children.end(), [](Node* lhs, Node* rhs) { return lhs->n > rhs->n; });

        for (auto& n : tree.root->children) {
            cout << n->debug(&env) << "\n";
        }

        int action = tree.pick();
        cout << "picking move " << env.decode(action).NaturalOut(&env.board) << "\n";
        tree.push(action);
    }

    cout << desc << ", " << value << "\n";
    cout << "final: \n" << env.print() << "\n";

    return 0;
}
