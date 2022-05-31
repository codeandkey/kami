#include "selfplay.h"
#include "env.h"

#include <iostream>

using namespace kami;
using namespace std;

int main() {
    cout << "Starting kami." << endl;

    NN model(8, 8, NFEATURES, PSIZE);

    Selfplay s(&model, 16, 512);
    s.start();

    while (1)
    {
        string inp;

        cout << "> ";
        getline(cin, inp);

        if (inp == "quit") break;
    }

    s.stop();

    return 0;
}
