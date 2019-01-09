// membership functions

#pragma once

#include "types.h"

namespace mf
{

// s-function

template <class T>
T s(T x, T a, T b, T c)
{
    if (x <= a)
        return 0.0;
    auto ca = c - a;
    if (a <= x && x <= b)
        return 2.0 * pow((x - a) / ca, 2);
    else if (b <= x && x <= c)
        return 1.0 - 2.0 * pow((x - c) / ca, 2);
    else
        return 1.0;
}

template <class T>
Array s(const Array &x, T a, T b, T c)
{
    auto sz = x.size();
    Array r(sz);
    for (auto i = 0; i < sz; i++)
        r(i) = s(x(i), a, b, c);
    return r;
}

template <class T>
T pi(T x, T b, T c)
{
    if (x <= c)
        return s(x, c - b, c - b / (T)2.0, c);
    else
        return 1.0 - s(x, c, c + b / (T)2.0, c + b);
}

template <class T>
Array pi(const Array &x, T b, T c)
{
    auto sz = x.size();
    Array r(sz);
    for (auto i = 0; i < sz; i++)
        r(i) = pi(x(i), b, c);
    return r;
}

// z-function

template <class T>
T z(T x, T a, T b, T c)
{
    return 1.0 - s(x, a, b, c);
}

template <class T>
Array z(const Array &x, T a, T b, T c)
{
    auto sz = x.size();
    Array r(sz);
    for (auto i = 0; i < sz; i++)
        r(i) = z(x(i), a, b, c);
    return r;
}

// ?

template <class T>
T g(T x, T a, T b)
{
    if (x <= a)
        return 0.0;
    if (a <= x && x <= b)
        return (x - a) / (b - a);
    else
        return 1.0;
}

template <class T>
Array s(const Array &x, T a, T b)
{
    auto sz = x.size();
    Array r(sz);
    for (auto i = 0; i < sz; i++)
        r(i) = g(x(i), a, b);
    return r;
}

// ?

template <class T>
T l(T x, T a, T b)
{
    if (x <= a)
        return 1;
    if (a <= x && x <= b)
        return (b - x) / (b - a);
    else
        return 0.0;
}

template <class T>
Array l(const Array &x, T a, T b)
{
    auto sz = x.size();
    Array r(sz);
    for (auto i = 0; i < sz; i++)
        r(i) = l(x(i), a, b);
    return r;
}

// triangle function

template <class T>
T t(T x, T a, T b, T c)
{
    if (x <= a)
        return 0.0;
    if (a <= x && x <= b)
        return (x - a) / (b - a);
    if (b <= x && x <= c)
        return (c - x) / (c - b);
    else
        return 0.0;
}

template <class T>
Array t(const Array &x, T a, T b, T c)
{
    auto sz = x.size();
    Array r(sz);
    for (auto i = 0; i < sz; i++)
        r(i) = t(x(i), a, b, c);
    return r;
}

template <class T>
T bell(T x, T a, T b, T c)
{
    return 1.0 / (1 + pow(fabs((x - c) / a), 2 * b));
}

template <class T>
auto bell(const Array &x, T a, T b, T c)
{
    return 1.0 / (1.0 + (((x - c) / a)).abs().pow(2*b));
}

template <class A, class T>
A gauss(const A &x, T sig, T c)
{
    return exp(- (x - c) * (x - c) / (2.0 * sig * sig));
}

}
