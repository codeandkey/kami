#pragma once

#include <string>

namespace kami::options {
    int getInt(std::string key, int def=0);
    std::string getStr(std::string key, std::string def="");

    void setInt(std::string key, int value);
    void setStr(std::string key, std::string value);

    void load(std::string path="options.yml");
    void write(std::string path="options.yml");

    void print();
}
