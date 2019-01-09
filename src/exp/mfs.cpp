#include "fuzzy.h"
#include "mf.h"
#include "plot.h"

#include <primitives/executor.h>

#include <iostream>

int main()
try
{
    auto print_mf = [](const auto &d)
    {
        auto x = seq<double>(d.xmin, d.xmax, .1);
        decltype(x) y;
        for (auto &v : x)
            y.push_back(d.f(v, d.a, d.b, d.c));

        python::Figure p;
        p.plots(0, 0).title = d.plot_name;
        p.addSeries(Series{ x,y });
        p.save("mf_"s + d.name);
    };

    struct mf_desc
    {
        String name;
        String plot_name;
        double a, b, c;
        double xmin, xmax;
        std::function<double(double, double, double, double)> f;
    };

    mf_desc mfs[] = {
        {"s", "", 0, 15, 30, -5, 35,
         [](auto v, auto a, auto b, auto c) { return mf::s<double>(v, a, b, c); }},
        {"z", "", 0, 15, 30, -5, 35,
         [](auto v, auto a, auto b, auto c) { return mf::z<double>(v, a, b, c); }},
        {"pi", "", 0, 15, 30, 10, 50,
         [](auto v, auto a, auto b, auto c) { return mf::pi<double>(v, b, c); }},
        {"g", "", 0, 15, 30, -5, 20,
         [](auto v, auto a, auto b, auto c) { return mf::g<double>(v, a, b); }},
        {"l", "", 0, 15, 30, -5, 20,
         [](auto v, auto a, auto b, auto c) { return mf::l<double>(v, a, b); }},
        {"t", "", 0, 15, 30, -5, 35,
         [](auto v, auto a, auto b, auto c) { return mf::t<double>(v, a, b, c); }},
        {"bell", "", 2, 4, 6, 2, 10,
         [](auto v, auto a, auto b, auto c) { return mf::bell<double>(v, a, b, c); }},
        {"gauss", "", 3, 5, 0, -5, 15,
         [](auto v, auto a, auto b, auto c) { return mf::gauss<double>(v, a, b); }},
    };

    for (auto &mf : mfs)
        print_mf(mf);

    FuzzyNumber f1;
    f1.n = mf::t<double>(vec(10, 50), 0, 15, 30);
    auto f2 = f1;
    f1.print("f1_before");
    (f1 + f2).print("f1_after_add");
    (f1 - f2).print("f1_after_add_div2");
    (f1 * f2).print("f1_after_mult");

    getExecutor().wait();

    return 0;
}
catch (const std::exception &e)
{
    std::cerr << e.what() << "\n";
    return 1;
}
catch (...)
{
    std::cerr << "Unhandled unknown exception" << "\n";
    return 1;
}

