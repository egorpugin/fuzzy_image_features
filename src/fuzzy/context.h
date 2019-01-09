#pragma once

#include <types.h>

#include <primitives/context.h>
#include <primitives/filesystem.h>

struct MatlabContext : primitives::Context
{
    MatlabContext()
    {
        addLine("close all;");
        emptyLines(1);
    }

    void addVariable(const String &n, const String &v)
    {
        addLine(n + " = " + v + ";");
    }

    template <class T>
    void addVector(const String &n, const std::vector<T> &v)
    {
        String s;
        s += "[";
        for (auto &e : v)
            s += std::to_string(e) + " ";
        s += "]";
        addVariable(n, s);
    }

    void addToPlot(const String &s, const String &n = String())
    {
        plots[n].addText(s + ",");
    }

    void beginPlot(const String &n = String())
    {
        //plots[n].addLine("fig = figure;");
        plots[n].addLine("figure;");
        plots[n].addLine("plot(");
    }

    void endPlot(const String &n = String())
    {
        plots[n].trimEnd(1);
        plots[n].addText(");");
        emptyLines();
        addLine(plots[n].getText());
        plots[n].clear();
    }

private:
    std::map<String, primitives::Context> plots;
};

struct PythonContext : primitives::Context
{
    PythonContext();

    template <class T>
    void addVector(const std::vector<T> &v, const String &name = String())
    {
        if (!name.empty())
            addLine(name + " = ");

        addText("[");
        auto sz = v.size();
        for (auto &e : v)
        {
            addText(std::to_string(e));
            addText(", ");
        }
        addText("]");
    }
};
