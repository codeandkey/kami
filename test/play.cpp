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
    float* lmm = new float[PSIZE];
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
                    legal_moves += ' ' + e.decode(a).NaturalOut(&e.board);

                while (1) {
                    string mv;

                    cout << "Your move: ";
                    cout.flush();
                    cin >> mv;

                    // Try decoding player move
                    thc::Move m;
                    bool ok = m.NaturalIn(&e.board, mv.c_str());

                    if (!ok)
                    {
                        cout << "Invalid move\n";
                        cout << "Legal moves:" << legal_moves;

                        continue;
                    }

                    // Advance tree.
                    // If the tree does not have children we must expand at the root level.

                    if (!tree.root->children.size())
                    {
                        if (!tree.select(obs, lmm))
                            throw runtime_error("expected tree to have children, can't expand for model!");

                        model.infer(obs, lmm, 1, inf_policy, &inf_value);
                        tree.expand(inf_policy, inf_value);
                    }

                    tree.push(e.encode(m));
                    break;
                }
            } else {
                cout << "Computer to move. Searching over " << nodes << " nodes." << endl;

                while (tree.n() < nodes)
                if (tree.select(obs, lmm))
                {
                    model.infer(obs, lmm, 1, inf_policy, &inf_value);
                    tree.expand(inf_policy, inf_value);
                }

                int picked = tree.pick();

                cout << "NN picks: " << e.decode(picked).NaturalOut(&e.board) << endl;
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
    delete[] lmm;

    cout << "Quitting. Final score " << score << "/" << game << endl;
    return 0;
}
