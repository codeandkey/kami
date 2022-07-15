#include "options.h"

#include <map>
#include <mutex>
#include <iostream>
#include <fstream>

#include <cerrno>
#include <cstring>

using namespace kami;
using namespace std;

static map<string, string> values;
static mutex values_lock;

void options::setInt(std::string key, int value)
{
    lock_guard<mutex> lock(values_lock);
    values[key] = to_string(value);
}

void options::setFloat(std::string key, float value)
{
    lock_guard<mutex> lock(values_lock);
    values[key] = to_string(value);
}

void options::setStr(string key, string value)
{
    lock_guard<mutex> lock(values_lock);
    values[key] = value;
}

string options::getStr(string key, string def)
{
    try {
        lock_guard<mutex> lock(values_lock);
        return values.at(key);
    } catch (out_of_range) {
        return def;
    }
}

int options::getInt(string key, int def)
{
    string sval;

    try {
        sval = getStr(key, to_string(def));
        return stoi(sval);
    } catch (exception& e) {
        throw runtime_error(string("conversion failure for key \"") + key + "\" = \"" + sval + "\": " + e.what());
    }
}

float options::getFloat(string key, float def)
{
    string sval;

    try {
        sval = getStr(key, to_string(def));
        return stof(sval);
    } catch (exception& e) {
        throw runtime_error(string("conversion failure for key \"") + key + "\" = \"" + sval + "\": " + e.what());
    }
}

void options::write(string path)
{
    lock_guard<mutex> lock(values_lock);
    ofstream file(path);

    if (!file)
        throw runtime_error("couldn't open " + path + " for writing: " + strerror(errno));

    for (auto& p : values) 
        file << p.first << ": " << p.second << endl;
}

void options::print()
{
    lock_guard<mutex> lock(values_lock);

    for (auto& p : values) 
        cout << p.first << ": " << p.second << endl;
}

void options::load(string path)
{
    lock_guard<mutex> lock(values_lock);
    ifstream file(path);

    if (!file)
        throw runtime_error("couldn't open " + path + " for reading: " + strerror(errno));

    string nextline;
    int line = 0;

    auto trim = [](string& s) {
        // trim leading whitespace
        while (s.size() && isspace(s[0]))
            s.erase(s.begin(), s.begin() + 1);

        // trim trailing whitespace
        while (s.size() && isspace(s[s.size() - 1]))
            s.pop_back();
    };

    cout << "Importing options from " << path << endl;

    while (getline(file, nextline))
    {
        ++line;
        
        // trim comments
        int p = nextline.find('#');

        if (p != string::npos)
            nextline.erase(nextline.begin() + p, nextline.end());

        // check key
        p = nextline.find(':');

        if (p == string::npos)
            continue;

        if (!p)
            throw runtime_error("invalid key at " + path + ":" + to_string(line));

        string key = nextline.substr(0, p);
        string val = nextline.substr(p + 1);

        trim(key);
        trim(val);

        if (!key.size() || !val.size())
            throw runtime_error("invalid option at " + path + ":" + to_string(line));

        cout << key << ": " << val << endl;
        values[key] = val;
    }
}
