#pragma once

#include <common.h>
#include <context.h>
#include <types.h>

#include <optional>

enum class Language
{
    Matlab,
    Python,
};

struct Script
{
    path print() const;
    void run(path p = path()) const;

protected:
    virtual path getExtension() const = 0;
    virtual Strings getArgs(const path &p) const = 0;
    virtual const primitives::Context &getContext() const = 0;
};

template <class Ctx>
struct ScriptData : Script
{
protected:
    mutable Ctx ctx;

    const primitives::Context &getContext() const override { return ctx; }
};

struct Series
{
    VectorD x, y;
};

struct SeriesArray
{
    Array x, y;
};

namespace python
{

struct Script : ::ScriptData<PythonContext>
{
private:
    path getExtension() const override;
    Strings getArgs(const path &p) const override;
};

/*template <class T>
struct SeriesBase
{
    T x;
    T y;*/

    /*SeriesBase() = default;

    SeriesBase(const SeriesBase &s)
        : x(s.x), y(s.y)
    {
    }*/
/*};

struct Series : SeriesBase<std::vector<double>>
{*/
    //using SeriesBase<std::vector<double>>::SeriesBase;

    /*template <class U>
    Series(const SeriesBase<U> &s)
    {
        auto conv = [](auto &a, auto &v)
        {
            auto sz = a.size();
            v.reserve(sz);
            for (int i = 0; i < sz; i++)
                v.push_back(a(i));
        };

        conv(s.x, x);
        conv(s.y, y);
    }*/
//};

struct Axis
{
    String label;
    std::optional<double> min;
    std::optional<double> max;
};

struct Plot
{
    String title;
    Axis x;
    Axis y;
    std::vector<Series> series;
    bool legend = false;
    bool colored = true;

    void emit(PythonContext &ctx) const;
};

struct Figure : Script
{
    Eigen::Array<Plot, -1, -1> plots;

    Figure();

    void addSeries(const Array &s);
    void addSeries(const Series &s);
    void addSeries(const SeriesArray &s);
    void addSeries(int row, int col, const Series &s);
    void addSeries(int row, int col, const SeriesArray &s);

    void show() const;
    void save(const path &p) const { save1(getOutputDir() / p); }

private:
    mutable bool emitted = false;

    void emit() const;
    void save1(const path &p) const;
};

}
