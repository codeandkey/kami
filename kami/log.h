#pragma once

#include <iostream>
#include <iomanip>
#include <ctime>

namespace kami {
    std::ostream& write_log(const char* tag) {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        std::cerr << std::setw(14) << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << " " << tag << " > ";
        return std::cerr;
    }
}

#define warn kami::write_log("W")
#define info kami::write_log("I")
#define err  kami::write_log("E")
