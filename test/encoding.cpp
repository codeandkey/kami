#include "../kami/env.h"

#include <iostream>
#include <string>
#include <stdexcept>

using namespace kami;
using namespace std;

int main()
{
    // Not every action is valid at every state (can't be encoded), so we run
    // an environment from start to finish, taking random actions at each step.
    // At every step we verify that each encoded action decodes to the same action.

    Env e;
    float val;

    cout << "Starting action test" << endl;
    while (!e.terminal(&val))
    {
        vector<int> actions = e.actions();

        for (int a : actions)
        {
            ncMove decoded = e.decode(a);
            char uci[6];
            ncMoveUCI(decoded, uci);
            int recoded = e.encode(decoded);
            if (a != recoded)
            {
                throw runtime_error("encoding failed for action " + to_string(a) + " , decode -> " + string(uci) + ", recode -> " + to_string(recoded));
            }
        }

        int picked = actions[rand() % actions.size()];
        ncMove decoded = e.decode(picked);
        char uci[6];
        ncMoveUCI(decoded, uci);
        cout << "Pushing " << uci << endl;
        e.push(picked);
    }

    cout << "Done" << endl;
    return 0;
}
