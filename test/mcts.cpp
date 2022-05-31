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
    MCTS tree;
    float value;
    std::string desc;

    srand(time(NULL));

    while (!tree.get_env().terminal_str(&value, desc))
    {
        cout << "==============================\n";
        cout << tree.get_env().print() << "\n";
        cout << "legal actions: ";

        for (int i : tree.get_env().actions()) 
        {
            if (tree.get_env().decode(i) != tree.get_env().decode(tree.get_env().encode(tree.get_env().decode(i))))
            {
                cout << i << " failed, decodes to " << tree.get_env().decode(i).TerseOut() << "\n";
                cout << ", but " << tree.get_env().decode(i).TerseOut() << " encodes to " << tree.get_env().encode(tree.get_env().decode(i)) << "\n";
            }

            cout << " " << tree.get_env().decode(i).NaturalOut(&tree.get_env().board);
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
            float p = 1.0f / (float) tree.get_env().actions().size();

            for (int i = 0; i < 4096 + 4 * 22; ++i)
                policy[i] = p;

            double value = (((double) rand() / (double) RAND_MAX) * 2.0 - 1.0);

            tree.expand(policy, value);
        }

        std::sort(tree.root->children.begin(), tree.root->children.end(), [](Node* lhs, Node* rhs) { return lhs->n > rhs->n; });

        for (auto& n : tree.root->children) {
            cout << n->debug(&tree.get_env()) << "\n";
        }

        int action = tree.pick();
        cout << "picking move " << tree.get_env().decode(action).NaturalOut(&tree.get_env().board) << "\n";
        tree.push(action);
    }

    cout << desc << ", " << value << "\n";
    cout << "final: \n" << tree.get_env().print() << "\n";

    return 0;
}
