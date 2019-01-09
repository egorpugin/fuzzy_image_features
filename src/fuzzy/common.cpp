#include "common.h"

path getMutablePath()
{
    auto mkname = [](int i) -> String
    {
        char buf[10] = { 0 };
        snprintf(buf, 10, "%04d", i);
        return buf;
    };

    static const auto run = [&]()
    {
        int i = 1;
        while (fs::exists(getOutputDir() / mkname(i)))
            i++;
        return i;
    }();
    return getOutputDir() / mkname(run);
}
