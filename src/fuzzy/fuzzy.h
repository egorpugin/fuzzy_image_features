#pragma once

#include "common.h"
#include "mf.h"

//template <class T>
struct FuzzyNumber
{
    Array n;

    void print(const String &name) const;

    FuzzyNumber &operator+=(const FuzzyNumber &rhs)
    {
        n = n + rhs.n - n * rhs.n;
        return *this;
    }

    FuzzyNumber &operator-=(const FuzzyNumber &rhs)
    {
        n = (n + rhs.n) / 2.0;
        return *this;
    }

    FuzzyNumber &operator*=(const FuzzyNumber &rhs)
    {
        n = n * rhs.n;
        return *this;
    }
};

inline FuzzyNumber operator+(const FuzzyNumber &lhs, const FuzzyNumber &rhs)
{
    auto f1 = lhs;
    f1 += rhs;
    return f1;
}

inline FuzzyNumber operator-(const FuzzyNumber &lhs, const FuzzyNumber &rhs)
{
    auto f1 = lhs;
    f1 -= rhs;
    return f1;
}

inline FuzzyNumber operator*(const FuzzyNumber &lhs, const FuzzyNumber &rhs)
{
    auto f1 = lhs;
    f1 *= rhs;
    return f1;
}
