
#include <iostream>
#include <cstdlib>

#define ASSERT(condition, msg) \
    do \
    { \
        if (!(condition)) \
        { \
            std::clog << "Fataler Fehler: " << msg << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)
