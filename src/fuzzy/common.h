#pragma once

#include "types.h"

#include <primitives/filesystem.h>

#define PIXEL(m, t, x, y) m.at<t>((x), (y))
#define PIXEL_8(m, x, y) PIXEL((m), uint8_t, (x), (y))
#define PIXEL_F(m, x, y) PIXEL((m), float, (x), (y))
#define PIXEL_D(m, x, y) PIXEL((m), double, (x), (y))

template <typename T>
auto seq(T begin, T end, T step = 1)
{
    std::vector<T> v;
    v.reserve((int)((end - begin) / step) + 1);
    for (T i = begin; i <= end; i += step)
        v.push_back(i);
    return v;
}

template <typename T>
auto vec(T begin, T end, T step = 1)
{
    Array v;
    v.resize((int)((end - begin) / step) + 1);
    int n = 0;
    for (T i = begin; i <= end; i += step, n++)
        v(n) = i;
    return v;
}

path getOutputDir();
path getMutablePath();
