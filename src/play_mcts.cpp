#include <iostream>

#include "mcts.h"
#include "env.h"

using namespace kami;
using namespace std;

int main()
{
    Env env;
    MCTS tree(&env);

    float value;

    while (!env.terminal(&value))
    {
        cout << env.print() << endl;
        cout << "legal actions: ";
        for (int i : env.actions()) 
        {
            if (env.decode(i) != env.decode(env.encode(env.decode(i))))
            {
                cout << i << " failed, decodes to " << env.decode(i).TerseOut() << endl;
                cout << ", but " << env.decode(i).TerseOut() << " encodes to " << env.encode(env.decode(i)) << endl;
            }

            cout << " " << env.decode(i).TerseOut();
        }
        cout << endl;

        while (tree.n() < 1024)
        {
            float observation[8 * 8 * NFEATURES];

            while (!tree.select(observation) && tree.n() < 1024);
            if (tree.n() >= 1024) break;
/*
            cout << "observing square A1:";

            for (int i = 0; i < NFEATURES; ++i)
                cout << " " << (int) observation[i];

            cout << endl;
*/
            float policy[4096 + 4 * 22];

            for (int i = 0; i < 4096 + 4 * 22; ++i)
                policy[i] = 1.0f / (float) (4096 + 4 * 22);

            tree.expand(policy, 0.0f);
        }

        int action = tree.pick();
        cout << "picking move " << env.decode(action).TerseOut() << endl;
        tree.push(action);
        cout << "terminal state is now " << env.terminal(&value) << endl;
    }

    cout << "game over, value = " << value << endl;
    return 0;
}
