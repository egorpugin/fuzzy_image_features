#include "plot.h"

#include <boost/algorithm/string.hpp>
#include <primitives/command.h>
#include <primitives/executor.h>

String python_string(const String &s)
{
    return "r'" + s + "'";
}

path Script::print() const
{
    auto p = getOutputDir() / "scripts" / unique_path();
    p += getExtension();
    write_file(p, getContext().getText());
    return p;
}

void Script::run(path p) const
{
    if (p.empty())
        p = print();

    primitives::Command c;
    c.args = getArgs(p);

    getExecutor().push([c]() mutable
    {
        auto make_error_string = [&c](const String &e)
        {
            String s = e;
            if (!c.out.text.empty())
                s += "\n" + boost::trim_copy(c.out.text);
            if (!c.err.text.empty())
                s += "\n" + boost::trim_copy(c.err.text);
            s += "\n";
            boost::trim(s);
            return s;
        };

        try
        {
            c.execute();
        }
        catch (std::exception &e)
        {
            throw std::runtime_error(make_error_string(e.what()));
        }
    });
}

namespace python
{

Strings Script::getArgs(const path &p) const
{
    return { "python", p.string() };
}

path Script::getExtension() const
{
    return ".py";
}

void Plot::emit(PythonContext &ctx) const
{
    for (auto &s : series)
    {
        if (!s.x.empty())
            ctx.addVector(s.x, "x");
        if (!s.y.empty())
            ctx.addVector(s.y, "y");
        if (!s.x.empty() && !s.y.empty())
            ctx.addLine("ax.plot(x, y)");
        else if (!s.y.empty())
            ctx.addLine("ax.plot(y)");
        else
            ctx.addLine("ax.plot(x)");
    }

    if (!title.empty())
        ctx.addLine("ax.set_title(" + python_string(title) + ")");

    if (!x.label.empty())
        ctx.addLine("ax.xlabel(" + python_string(x.label) + ")");
    if (!y.label.empty())
        ctx.addLine("ax.ylabel(" + python_string(y.label) + ")");

    auto add_limit = [&ctx](const auto &v, const auto &s)
    {
        if (v)
            ctx.addLine("ax.set_xlim(" + s + " = " + std::to_string(v.value()) + ")");
    };

    auto add_limits = [&add_limit](const auto &v, const auto &s)
    {
        add_limit(v.min, s + "min"s);
        add_limit(v.max, s + "max"s);
    };

    add_limits(x, "x");
    add_limits(y, "y");

    if (legend)
        ctx.addLine("ax.legend()");

    ctx.emptyLines();
}

Figure::Figure()
{
    ctx.addLine(R"(
# tricky imports
from __future__ import unicode_literals

# basic imports
#import os

# math impots
import matplotlib
matplotlib.use("pgf")

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

plt.ioff()
)");

    plots.resize(1, 1);
}

void Figure::addSeries(const Array & s)
{
    addSeries(SeriesArray{Array(), s});
}

void Figure::addSeries(const Series &s)
{
    addSeries(0, 0, s);
}

void Figure::addSeries(const SeriesArray & s)
{
    addSeries(0, 0, s);
}

void Figure::addSeries(int row, int col, const Series &s)
{
    plots(row, col).series.push_back(s);
}

void Figure::addSeries(int row, int col, const SeriesArray & s)
{
    auto conv = [](auto &a, auto &v)
    {
        auto sz = a.size();
        v.reserve(sz);
        for (int i = 0; i < sz; i++)
            v.push_back(a(i));
    };

    Series s2;
    conv(s.x, s2.x);
    conv(s.y, s2.y);

    addSeries(row, col, s2);
}

void Figure::emit() const
{
    if (emitted)
        return;
    emitted = true;

    ctx.addLine("fig = plt.figure()");
    ctx.emptyLines();

    int np = 1;
    for (int r = 0; r < plots.rows(); r++)
    {
        for (int c = 0; c < plots.cols(); c++)
        {
            ctx.addLine("ax = fig.add_subplot(" + std::to_string(plots.rows()) + ", " + std::to_string(plots.cols()) + ", " + std::to_string(np) + ")");
            plots(r, c).emit(ctx);
            ctx.emptyLines();
        }
    }
}

void Figure::show() const
{
    emit();
    auto p = print();
    append_file(p, "plt.show()");
    run(p);
}

void Figure::save1(const path &p) const
{
    emit();
    auto p2 = print();
    append_file(p2, "plt.savefig('" + normalize_path(p) + ".pdf')");
    run(p2);
}

}
