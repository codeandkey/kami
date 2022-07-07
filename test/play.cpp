#include "../src/mcts.h"
#include "../src/env.h"
#include "../src/nn/nn.h"

#include <iostream>
#include <string>

using namespace kami;
using namespace std;

int main(int argc, char** argv)
{
    float score = 0.0f;
    int game;
    const int nodes = 1024;

    NN model(8, 8, NFEATURES, PSIZE);
    MCTS tree;

    if (argc > 1)
    {
        cout << "Loading model from " << argv[1] << endl;
        model.read(argv[1]);
    }

    float* obs = new float[8 * 8 * NFEATURES];
    float* inf_policy = new float[PSIZE];
    float inf_value;

    for (game = 1;; ++game)
    {
        string choice;
        Env& e = tree.get_env();

        int pturn = (rand() % 2) * 2 - 1;
        float value;

        while (!e.terminal(&value))
        {
            cout << e.print() << endl;

            if (e.turn() == pturn) {
                string legal_moves;
                vector<int> legal_actions = e.actions();

                for (int a : legal_actions)
                    legal_moves += ' ' + e.debug_action(a);

                while (1) {
                    string mv;

                    cout << "Your move: ";
                    cout.flush();
                    cin >> mv;

                    // Try decoding player move
                    int selected = -1;
                    ncMove wants = ncMoveFromUci((char*) mv.c_str());

                    for (auto& a : legal_actions)
                    if (tree.get_env().decode(a) == wants)
                    {
                        selected = a;
                        break;
                    }

                    if (selected < 0)
                    {
                        cout << "Invalid move\n";
                        cout << "Legal moves:" << legal_moves;

                        continue;
                    }

                    // Advance tree.
                    // If the tree does not have children we must expand at the root level.

                    if (!tree.root->children.size())
                    {
                        if (!tree.select(obs))
                            throw runtime_error("expected tree to have children, can't expand for model!");

                        model.infer(obs, 1, inf_policy, &inf_value);
                        tree.expand(inf_policy, inf_value);
                    }

                    tree.push(selected);
                    break;
                }
            } else {
                cout << "Computer to move. Searching over " << nodes << " nodes." << endl;

                while (tree.n() < nodes)
                if (tree.select(obs))
                {
                    model.infer(obs, 1, inf_policy, &inf_value);
                    tree.expand(inf_policy, inf_value);
                }

                int picked = tree.pick();

                cout << "NN picks: " << e.debug_action(picked) << endl;
                tree.push(picked);
            }
        }

        if (value == pturn)
            score += 1.0f;
        else if (value == 0.0f)
            score += 0.5f;

        string resp;
        cout << "Score: " << score << "/" << game << "\nContinue? (Y/n)";
        cout.flush();
        getline(cin, resp);

        if (tolower(resp[0]) != 'Y')
            break;

        tree.reset();
    }

    delete[] inf_policy;
    delete[] obs;

    cout << "Quitting. Final score " << score << "/" << game << endl;
    return 0;
}
