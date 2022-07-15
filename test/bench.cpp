#include <algorithm>
#include <iostream>
#include <iomanip>

#include <cmath>
#include <ctime>

#include "../kami/mcts.h"
#include "../kami/env.h"

using namespace kami;
using namespace std;

int main()
{
    MCTS tree;

    float value;
    srand(time(NULL));

    while (!tree.get_env().terminal(&value))
    {
        clock_t tm = clock();
        long observations = 0;

        while (tree.n() < 1024)
        {
            float observation[8 * 8 * NFEATURES];

            while (!tree.select(observation) && tree.n() < 1024);
            if (tree.n() >= 1024) break;

            observations++;

            float policy[4096 + 4 * 22];
            float p = 1.0f / (float) tree.get_env().actions().size();

            for (int i = 0; i < 4096 + 4 * 22; ++i)
                policy[i] = p;

            double value = (((double) rand() / (double) RAND_MAX) * 2.0 - 1.0);

            tree.expand(policy, value);
        }

        cout << "\rObservations / second: " << (observations * CLOCKS_PER_SEC) / (clock() - tm);

        tree.push(tree.pick());
    }

    cout << "\n";
    return 0;
}
