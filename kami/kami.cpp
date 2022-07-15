#include "selfplay.h"
#include "env.h"
#include "options.h"

#include <ctime>
#include <iostream>
#include <fstream>
#include <functional>
#include <map>

using namespace kami;
using namespace std;

int main(int argc, char** argv) {
    cout << "Starting kami." << endl;

    // Try and load options
    try {
        options::load();
    } catch (exception& e) {
        cerr << "WARNING: " << e.what() << endl;
    }

    cout << "=========== OPTIONS ===========" << endl;
    options::print();
    cout << "========= END OPTIONS =========" << endl;

    if (options::getInt("force_torch_single_threaded", 0))
    {
        // Force single-threaded torch (we do the multithreading around here!)
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
    }

    srand(time(NULL));

    NN model(8, 8, NFEATURES, PSIZE);

    string modelpath = options::getStr("model_path");

    if (modelpath.size())
    {
        cout << "Restoring model from " << modelpath << endl;

        try {
            model.read(modelpath);
            cout << "Loaded model." << endl;
        } catch (exception& e) {
            cerr << "WARNING: model read from " << modelpath << " failed: " << e.what() << endl;
        }
    }

    Selfplay s(&model);
    s.start();

    bool should_quit = false;
    map<string, function<void(vector<string>& args)>> commands;

    commands["help"] = [&](vector<string>& args) { cout << "No help for you!" << endl; };

    commands["write"] = [&](vector<string>& args)
    {
        string path;

        if (!args.size())
            args.push_back(modelpath);

        for (int i = 0; i < args.size(); ++i)
        {
            path += args[i];

            if (i + 1 != args.size())
                path += ' ';
        }

        cout << "Saving model to " << path << "...";
        cout.flush();

        try {
            model.write(path);
        } catch (exception& e)
        {
            cerr << "model.write() ERROR: " << e.what() << endl;
            return;
        }

        cout << "done" << endl;
    };

    commands["read"] = [&](vector<string>& args)
    {
        string path;

        if (!args.size())
            args.push_back(modelpath);

        for (int i = 0; i < args.size(); ++i)
        {
            path += args[i];

            if (i + 1 != args.size())
                path += ' ';
        }

        cout << "Reading model from " << path << "...";
        cout.flush();

        try {
            model.read(path);
        } catch (exception& e)
        {
            cerr << "model.read() ERROR: " << e.what() << endl;
            return;
        }

        cout << "done" << endl;
    };

    commands["pgn"] = [&](vector<string>& args)
    {
        string nextpgn;
        nextpgn = string("[White \"KAMI generation ") + to_string(model.get_generation()) + "\"]\n";
        nextpgn += string("[Black \"KAMI generation ") + to_string(model.get_generation()) + "\"]\n";
        nextpgn += s.get_next_pgn();

        cout << "\n" << nextpgn << "\n";

        if (args.size()) {
            ofstream file(args[0]);

            if (!file)
            {
                cerr << "error writing PGN to " << args[0] << "\n";
                return;
            }

            file << nextpgn << endl;
            file.close();

            cout << "Wrote PGN data to " << args[0] << endl;
        }
    };

    commands["status"] = [&](vector<string>& args)
    {
        cout << "Inference threads: " << options::getInt("inference_threads") << endl;
        cout << "Total experiences: " << s.get_rbuf().count() << endl;
        cout << "Current generation: " << model.get_generation() << endl;
    };

    string line;
    vector<string> args;

    while (1)
    {
        cout << "> ";
        cout.flush();
        if (!getline(cin, line)) break;

        line += '\0';

        for (char* tok = strtok(&line[0], " \t"); tok; tok = strtok(nullptr, " \t"))
            args.push_back(tok);

        if (!args.size())
            continue;

        try {
            string cmd = args[0];
            args.erase(args.begin(), args.begin() + 1);

            if (cmd == "quit")
            {
                cerr << "Quitting.." << endl;
                break;
            }


            try {
                commands[cmd](args);
            } catch (exception& e) {
                cerr << "Error in command: " << e.what() << endl;
            }
        } catch (out_of_range& e) {
            cerr << "Invalid command \"" << args[0] << "\"" << endl;
        }
    }

    s.stop();

    return 0;
}
