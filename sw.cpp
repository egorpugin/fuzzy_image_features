void build(Solution &s)
{
    auto &fuzzy = s.addStaticLibrary("fuzzy");
    fuzzy.CPPVersion = CPPLanguageStandard::CPP17;
    fuzzy.PackageDefinitions = true;
    fuzzy.Public += "src/output_dir.cpp";
    fuzzy += "src/fuzzy/.*"_rr;
    fuzzy.Public += "src/fuzzy"_idir;
    fuzzy.Public +=
        "org.sw.demo.intel.opencv.imgproc-*"_dep,
        "org.sw.demo.intel.opencv.highgui-*"_dep,
        "pub.egorpugin.primitives.filesystem-master"_dep,
        "pub.egorpugin.primitives.date_time-master"_dep,
        "pub.egorpugin.primitives.context-master"_dep,
        "pub.egorpugin.primitives.executor-master"_dep,
        "pub.egorpugin.primitives.command-master"_dep,
        "pub.egorpugin.primitives.log-master"_dep,
        "org.sw.demo.eigen-*"_dep,
        //"org.sw.demo.microsoft.range_v3_vs2015-master"_dep
        "org.sw.demo.ericniebler.range_v3-master"_dep
        ;

    auto &exp = s.addDirectory("exp");

    auto add_target = [&exp, &fuzzy](const String &name)
    {
        auto &t = exp.addExecutable(name);
        t.CPPVersion = CPPLanguageStandard::CPP17;
        t.PackageDefinitions = true;
        t += path("src/exp/" + name + ".cpp");
        t += fuzzy;
    };

    add_target("mfs");
    add_target("edges");
    add_target("binarization");
    add_target("filter");
}
