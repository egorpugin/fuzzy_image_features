#include <context.h>
#include <fuzzy.h>
#include <imgproc.h>

#include <boost/algorithm/string.hpp>
#include <opencv2/imgproc.hpp>
#include <primitives/executor.h>

#include <iostream>
#include <numeric>

#include <primitives/log.h>
DECLARE_STATIC_LOGGER(logger, "filter");

const std::string markers = "o+*.xsd^v><ph";

struct result_existing
{
    std::vector<double> psnr;
    std::vector<double> eqm;
};

struct result_new
{
    std::vector<double> lv_psnr, lv_eqm;
    std::map<double /*level*/, result_existing> sz_level_results;
};

struct result
{
    result_existing re;
    std::map<int /*sz*/, result_new> rn;
    std::map<double /*level*/, result_existing> rn_level;
};

auto mf1(f64 v)
{
    auto sz = 60.0;
    auto a = -sz + v;
    auto c = v;
    auto b = sz / 2.0;
    return [b, c](auto x) { return mf::pi(x, b, c); };
}

cv::Mat make_fuzzy_representation(const cv::Mat &ideal, const cv::Mat &current)
{
    static const std::map<int, std::function<double(double)>> mfs = []
    {
        std::map<int, std::function<double(double)>> mfs;
        for (int i = 0; i < 256; i++)
            mfs[i] = mf1(i);
        return mfs;
    }();

    cv::Mat fuzzy_representation;
    ideal.convertTo(fuzzy_representation, CV_64F);
    const auto s = fuzzy_representation.size();
    for (int i = 0; i < s.height; i++)
    {
        for (int j = 0; j < s.width; j++)
        {
            auto f = mfs.find(PIXEL_8(ideal, i, j))->second;
            PIXEL_D(fuzzy_representation, i, j) = f(PIXEL_8(current, i, j));
        }
    }
    return fuzzy_representation;
}

cv::Mat fuzzy_representation_filter(const cv::Mat &mask, cv::Mat &ideal, const cv::Mat &current)
{
    cv::Mat x = cv::Mat::zeros(ideal.size(), ideal.type());
    const auto s = mask.size();
    for (int i = 0; i < s.height; i++)
    {
        for (int j = 0; j < s.width; j++)
        {
            // if we accept current u, we take pixel from original (current) image
            if (PIXEL_8(mask, i, j))
                PIXEL_8(x, i, j) = PIXEL_8(current, i, j);
            // else we take it from ideal (prepared) image
            else
                PIXEL_8(x, i, j) = PIXEL_8(ideal, i, j);
        }
    }
    return x;
}

void hor(int sz, const cv::Mat &in, cv::Mat &out)
{
    in.convertTo(out, CV_32F);

    cv::Mat x = cv::Mat::zeros(out.size(), out.type());
    const auto s = out.size();
    const auto offset = sz / 2;
    for (int i = offset; i < s.height - offset; i++)
    {
        for (int j = offset; j < s.width - offset; j++)
        {
            float sum = 0;
            for (auto n = j - 1; n <= j + offset; n++)
                sum += PIXEL_F(out, i, n);
            for (auto n = j + 1; n <= j + offset; n++)
                sum += PIXEL_F(out, i, n);
            PIXEL_F(x, i, j) = sum / float(offset * 2);
        }
    }
    out = float_to_uint8(x);
}

void vert(int sz, const cv::Mat &in, cv::Mat &out)
{
    in.convertTo(out, CV_32F);

    cv::Mat x = cv::Mat::zeros(out.size(), out.type());
    const auto s = out.size();
    const auto offset = sz / 2;
    for (int i = offset; i < s.height - offset; i++)
    {
        for (int j = offset; j < s.width - offset; j++)
        {
            float sum = 0;
            for (auto m = i - 1; m <= i + offset; m++)
                sum += PIXEL_F(out, m, j);
            for (auto m = i + 1; m <= i + offset; m++)
                sum += PIXEL_F(out, m, j);
            PIXEL_F(x, i, j) = sum / float(offset * 2);
        }
    }
    out = float_to_uint8(x);
}

void d4(int sz, const cv::Mat &in, cv::Mat &out)
{
    in.convertTo(out, CV_32F);

    cv::Mat x = cv::Mat::zeros(out.size(), out.type());
    const auto s = out.size();
    const auto offset = sz / 2;
    for (int i = offset; i < s.height - offset; i++)
    {
        for (int j = offset; j < s.width - offset; j++)
        {
            float sum = 0;
            for (auto n = j - 1; n <= j + offset; n++)
                sum += PIXEL_F(out, i, n);
            for (auto n = j + 1; n <= j + offset; n++)
                sum += PIXEL_F(out, i, n);
            for (auto m = i - 1; m <= i + offset; m++)
                sum += PIXEL_F(out, m, j);
            for (auto m = i + 1; m <= i + offset; m++)
                sum += PIXEL_F(out, m, j);
            PIXEL_F(x, i, j) = sum / float(offset * 2 * 2);
        }
    }
    out = float_to_uint8(x);
}

void dd(int sz, const cv::Mat &in, cv::Mat &out)
{
    in.convertTo(out, CV_32F);

    cv::Mat x = cv::Mat::zeros(out.size(), out.type());
    const auto s = out.size();
    const auto offset = sz / 2;
    for (int i = offset; i < s.height - offset; i++)
    {
        for (int j = offset; j < s.width - offset; j++)
        {
            float sum = 0;
            for (auto p = 1; p <= offset; p++)
            {
                sum += PIXEL_F(out, i - p, j - p);
                sum += PIXEL_F(out, i - p, j + p);
                sum += PIXEL_F(out, i + p, j - p);
                sum += PIXEL_F(out, i + p, j + p);
            }
            PIXEL_F(x, i, j) = sum / float(offset * 2 * 2);
        }
    }
    out = float_to_uint8(x);
}

void d8(int sz, const cv::Mat &in, cv::Mat &out)
{
    in.convertTo(out, CV_32F);

    cv::Mat x = cv::Mat::zeros(out.size(), out.type());
    const auto s = out.size();
    const auto offset = sz / 2;
    for (int i = offset; i < s.height - offset; i++)
    {
        for (int j = offset; j < s.width - offset; j++)
        {
            float sum = 0;
            for (auto m = i - 1; m <= i + offset; m++)
            {
                for (auto n = j - 1; n <= j + offset; n++)
                    sum += PIXEL_F(out, m, n);
            }
            sum -= PIXEL_F(out, i, j);
            PIXEL_F(x, i, j) = sum / float(((offset * 2 + 1) * (offset * 2 + 1)) - 1);
        }
    }
    out = float_to_uint8(x);
}

void d9(int sz, const cv::Mat &in, cv::Mat &out)
{
    in.convertTo(out, CV_32F);

    cv::Mat x = cv::Mat::zeros(out.size(), out.type());
    const auto s = out.size();
    const auto offset = sz / 2;
    for (int i = offset; i < s.height - offset; i++)
    {
        for (int j = offset; j < s.width - offset; j++)
        {
            float sum = 0;
            for (auto m = i - 1; m <= i + offset; m++)
            {
                for (auto n = j - 1; n <= j + offset; n++)
                    sum += PIXEL_F(out, m, n);
            }
            PIXEL_F(x, i, j) = sum / float(((offset * 2 + 1) * (offset * 2 + 1)));
        }
    }
    out = float_to_uint8(x);
}

void blur(int sz, const cv::Mat &in, cv::Mat &out)
{
    cv::blur(in, out, cv::Size(sz, sz));
}

void gauss_blur(int sz, const cv::Mat &in, cv::Mat &out)
{
    cv::GaussianBlur(in, out, cv::Size(sz, sz), 0);
}

void med_blur(int sz, const cv::Mat &in, cv::Mat &out)
{
    medianBlur(in, out, sz);
}

void bilat_filter(int sz, const cv::Mat &in, cv::Mat &out)
{
    cv::bilateralFilter(in, out, sz, sz * 2, sz / 2);
}

template <class F>
auto cache(F f, cv::Mat &im, const String &name)
{
    path p = "cache";
    p /= name;
    //if (imexists(p))
    //    return im = imread(p);
    f();
    //imwrite(p, im);
    return im;
}

template <class F>
auto cached_write(F f, const String &name)
{
    path p = "cache";
    p /= name;
    //if (!imexists(p))
    //imwrite(p, f());
}

// data
const std::vector<std::pair<std::string /* method */, std::function<void(int /* ksize */, const cv::Mat & /* in */, cv::Mat & /* out */)>>> methods{
    /*{ "hor", hor },
    { "vert", vert },
    { "d4", d4 },
    { "dd", dd },
    { "d8", d8 },
    { "d9", d9 },*/
    { "blur", blur },
    { "gauss_blur", gauss_blur },
    { "med_blur", med_blur },
    //{ "bilat_filter", bilat_filter },
    // add more algs here
};

std::vector<double> levels{
    0.9999, 0.999, 0.99, 0.98, 0.965, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.1, 0.01,
};

const int sz_from = 3;
const int sz_to = 16;

Executor e;

void process_noise(const cv::Mat &im, const cv::Mat &noised, const String &noise_name)
{
    imwrite(noise_name, noised);
    {
        auto p = qm::psnr(im, noised);
        auto e = qm::eqm(im, noised);
        LOG_INFO(logger, "Processing noise: " << noise_name << " PSNR " << p << " MSE " << e);
    }

    std::vector<int> x;
    std::map<std::string, result> results;

    // preallocate map values
    for (auto &[m, f] : methods)
        results[m];

    // calc filter results of different algorithms
    for (int sz = sz_from; sz < sz_to; sz += 2)
    {
        x.push_back(sz);

        //auto s = qm::ssim(im, n, 3);
        //std::cout << "ssim = " << s << "\n";

        e.push([sz, &im, &results, &noised, &noise_name]
        {
            LOG_INFO(logger, "sz = " << sz);

            cv::Mat denoised;
            for (auto &[m, f] : methods)
            {
                auto f2 = f;
                cache([f = f2, sz, &noised, &denoised]() {
                    f(sz, noised, denoised);
                }, denoised, "blurred_" + std::to_string(sz) + "_" + m);
                auto p = qm::psnr(im, denoised);
                results[m].re.psnr.push_back(p);
                auto e = qm::eqm(im, denoised);
                results[m].re.eqm.push_back(e);
                //std::cout << "psnr = " << p << "\n";

                // find fuzzy repr
                auto fr = make_fuzzy_representation(denoised, noised);
                auto fr_fn = "fr_" + std::to_string(sz) + "_" + m;
                cached_write([&fr] {
                    return float_to_uint8(fr);
                }, fr_fn);

                for (auto &l : levels)
                {
                    auto ln = fr_fn + "_" + std::to_string(l);
                    boost::replace_all(ln, ".", "_");
                    cv::Mat mask = fr >= l;
                    cv::Mat ffilter_result;
                    cache([&mask, &denoised, &noised, &ffilter_result]() {
                        ffilter_result = fuzzy_representation_filter(mask, denoised, noised);
                    }, ffilter_result, ln);

                    auto lp = qm::psnr(im, ffilter_result);
                    results[m].rn[sz].sz_level_results[l].psnr.push_back(lp);
                    auto le = qm::eqm(im, ffilter_result);
                    results[m].rn[sz].sz_level_results[l].eqm.push_back(le);

                    imwrite(getMutablePath() / "img" / ("f_" + noise_name + "_" + ln), ffilter_result);

                    //auto diff = lp - p;
                    //if (lp > p && diff > 0.1)
                    //    LOG_INFO(logger, "hit! p = " << p << ", lp = " << lp << ", diff = " << diff);
                }

                imwrite(getMutablePath() / "img" / ("f_" + noise_name + "_" + fr_fn), denoised);
            }
        });
    }

    e.wait();

    // write and calc initial values
    MatlabContext ctx;
    ctx.addVector("x", x);
    ctx.addVector("l", levels);
    for (auto &[m, f] : methods)
    {
        ctx.addVector(m + "_psnr", results[m].re.psnr);
        ctx.addVector(m + "_eqm", results[m].re.eqm);

        for (int sz = sz_from; sz < sz_to; sz += 2)
        {
            auto ln = m + "_fr_" + std::to_string(sz);
            for (auto &l : levels)
            {
                results[m].rn[sz].lv_psnr.push_back(results[m].rn[sz].sz_level_results[l].psnr[0]);
                results[m].rn[sz].lv_eqm.push_back(results[m].rn[sz].sz_level_results[l].eqm[0]);

                results[m].rn_level[l].psnr.push_back(results[m].rn[sz].sz_level_results[l].psnr[0]);
                results[m].rn_level[l].eqm.push_back(results[m].rn[sz].sz_level_results[l].eqm[0]);
            }
            ctx.addVector(ln + "_psnr", results[m].rn[sz].lv_psnr);
            ctx.addVector(ln + "_eqm", results[m].rn[sz].lv_eqm);
        }

        for (auto &l : levels)
        {
            auto ln = m + "_fr_" + std::to_string(l);
            boost::replace_all(ln, ".", "_");
            ctx.addVector(ln + "_psnr", results[m].rn_level[l].psnr);
            ctx.addVector(ln + "_eqm", results[m].rn_level[l].eqm);
        }
    }

    // calc psnr of noise image to original
    //auto p = qm::psnr(im, n);
    //auto e = qm::eqm(im, n);
    //std::cout << "psnr = " << p << "\n";

    // general plots
    {
        // PSNR
        int m = 0;
        ctx.beginPlot();
        for (auto &[me, f] : methods)
        {
            ctx.addToPlot("x");
            ctx.addToPlot(me + "_psnr");
            ctx.addToPlot("'- "s + markers[m++] + "'");
        }
        ctx.endPlot();
        ctx.addLine("xlabel('kernel size')");
        ctx.addLine("ylabel('PSNR')");
        ctx.addLine("legend({");
        for (auto &[me, f] : methods)
            ctx.addText("'" + me + "',");
        ctx.addText("},'Interpreter','none');");
        ctx.addLine("%print('PSNR_all_" + noise_name + "','-deps');");
        ctx.addLine("%print('PSNR_all_" + noise_name + "_colored','-depsc');");
        ctx.emptyLines();

        // MSE
        m = 0;
        ctx.beginPlot();
        for (auto &[me, f] : methods)
        {
            ctx.addToPlot("x");
            ctx.addToPlot(me + "_eqm");
            ctx.addToPlot("'- "s + markers[m++] + "'");
        }
        ctx.endPlot();
        ctx.addLine("xlabel('kernel size')");
        ctx.addLine("ylabel('MSE')");
        ctx.addLine("legend({");
        for (auto &[me, f] : methods)
            ctx.addText("'" + me + "',");
        ctx.addText("},'Interpreter','none');");
        ctx.addLine("%print('MSE_all_" + noise_name + "','-deps');");
        ctx.addLine("%print('MSE_all_" + noise_name + "_colored','-depsc');");
        ctx.emptyLines();
    }

    // filter results
    for (auto &[me, f] : methods)
    {
        // PSNR(level) for N max sizes
        {
            std::multimap<double, int, std::greater<double>> sz_max;
            for (int sz = sz_from; sz < sz_to; sz += 2)
            {
                //auto sum = std::accumulate(results[me].rn[sz].lv_psnr.begin(), results[me].rn[sz].lv_psnr.end(), 0.0);
                //auto max = sum / results[me].rn[sz].lv_psnr.size();
                auto max = std::max_element(results[me].rn[sz].lv_psnr.begin(), results[me].rn[sz].lv_psnr.end());
                sz_max.insert({ *max, sz });
            }
            auto n = markers.size();
            auto i = sz_max.begin();
            while (n-- && i != sz_max.end())
                i = std::next(i);
            sz_max.erase(i, sz_max.end());

            // PSNR
            int m = 0;
            ctx.beginPlot();
            for (auto &[sum, sz] : sz_max)
            {
                ctx.addToPlot("l");
                auto ln = me + "_fr_" + std::to_string(sz);
                ctx.addToPlot(ln + "_psnr");
                ctx.addToPlot("'- "s + markers[m++] + "'");
            }
            ctx.endPlot();
            ctx.addLine("xlabel('level')");
            ctx.addLine("ylabel('PSNR')");
            ctx.addLine("title({'PSNR(level), method = " + me + "'},'Interpreter','none')");
            ctx.addLine("legend({");
            for (auto &[sum, sz] : sz_max)
                ctx.addText("'" + std::to_string(sz) + "',");
            ctx.addText("},'Interpreter','none');");
            ctx.addLine("%print('PSNR_" + me + "_sz_" + noise_name + "','-deps');");
            ctx.addLine("%print('PSNR_" + me + "_sz_" + noise_name + "_colored','-depsc');");
            ctx.emptyLines();

            // MSE
            m = 0;
            ctx.beginPlot();
            for (auto &[sum, sz] : sz_max)
            {
                ctx.addToPlot("l");
                auto ln = me + "_fr_" + std::to_string(sz);
                ctx.addToPlot(ln + "_eqm");
                ctx.addToPlot("'- "s + markers[m++] + "'");
            }
            ctx.endPlot();
            ctx.addLine("xlabel('level')");
            ctx.addLine("ylabel('MSE')");
            ctx.addLine("title({'MSE(level), method = " + me + "'},'Interpreter','none')");
            ctx.addLine("legend({");
            for (auto &[sum, sz] : sz_max)
                ctx.addText("'" + std::to_string(sz) + "',");
            ctx.addText("},'Interpreter','none');");
            ctx.addLine("%print('MSE_" + me + "_sz_" + noise_name + "','-deps');");
            ctx.addLine("%print('MSE_" + me + "_sz_" + noise_name + "_colored','-depsc');");
            ctx.emptyLines();
        }

        // PSNR(level) for N max levels
        {
            std::multimap<double, double, std::greater<double>> l_max;
            for (auto &l : levels)
            {
                //auto sum = std::accumulate(results[me].rn[sz].lv_psnr.begin(), results[me].rn[sz].lv_psnr.end(), 0.0);
                //auto max = sum / results[me].rn[sz].lv_psnr.size();
                auto max = std::max_element(results[me].rn_level[l].psnr.begin(), results[me].rn_level[l].psnr.end());
                l_max.insert({ *max, l });
            }
            auto n = markers.size() - 1; // left 1 for original line
            auto i = l_max.begin();
            while (n-- && i != l_max.end())
                i = std::next(i);
            l_max.erase(i, l_max.end());

            // PSNR
            int m = 0;
            ctx.beginPlot();
            {
                // original line
                ctx.addToPlot("x");
                ctx.addToPlot(me + "_psnr");
                ctx.addToPlot("'- "s + markers[m++] + "'");
            }
            for (auto &[sum, l] : l_max)
            {
                ctx.addToPlot("x");
                auto ln = me + "_fr_" + std::to_string(l);
                boost::replace_all(ln, ".", "_");
                ctx.addToPlot(ln + "_psnr");
                ctx.addToPlot("'- "s + markers[m++] + "'");
            }
            ctx.endPlot();
            ctx.addLine("xlabel('kernel size')");
            ctx.addLine("ylabel('PSNR')");
            ctx.addLine("title({'PSNR(kernel size), method = " + me + "'},'Interpreter','none')");
            ctx.addLine("legend({");
            ctx.addText("'orig',");
            for (auto &[sum, l] : l_max)
                ctx.addText("'" + std::to_string(l) + "',");
            ctx.addText("},'Interpreter','none');");
            ctx.addLine("%print('PSNR_" + me + "_l_" + noise_name + "','-deps');");
            ctx.addLine("%print('PSNR_" + me + "_l_" + noise_name + "_colored','-depsc');");
            ctx.emptyLines();

            // MSE
            m = 0;
            ctx.beginPlot();
            {
                // original line
                ctx.addToPlot("x");
                ctx.addToPlot(me + "_psnr");
                ctx.addToPlot("'- "s + markers[m++] + "'");
            }
            for (auto &[sum, l] : l_max)
            {
                ctx.addToPlot("x");
                auto ln = me + "_fr_" + std::to_string(l);
                boost::replace_all(ln, ".", "_");
                ctx.addToPlot(ln + "_eqm");
                ctx.addToPlot("'- "s + markers[m++] + "'");
            }
            ctx.endPlot();
            ctx.addLine("xlabel('kernel size')");
            ctx.addLine("ylabel('MSE')");
            ctx.addLine("title({'MSE(kernel size), method = " + me + "'},'Interpreter','none')");
            ctx.addLine("legend({");
            ctx.addText("'orig',");
            for (auto &[sum, l] : l_max)
                ctx.addText("'" + std::to_string(l) + "',");
            ctx.addText("},'Interpreter','none');");
            ctx.addLine("%print('MSE_" + me + "_l_" + noise_name + "','-deps');");
            ctx.addLine("%print('MSE_" + me + "_l_" + noise_name + "_colored','-depsc');");
            ctx.emptyLines();
        }
    }

    write_file(getMutablePath() / ("f_" + noise_name + ".m"), ctx.getText());
}

void process_image(const path &imname)
{
    auto mkname = [&imname](const auto &s)
    {
        return imname.stem().string() + "_" + s;
    };

    auto im = imread(imname);
    cv::Mat noised;

    for (auto &v : { 0.1, 0.01, 0.001, 0.0001 })
    {
        noised = noise::gauss_var(im, 0, v);
        auto n = "g_" + std::to_string(v);
        boost::replace_all(n, ".", "_");
        process_noise(im, noised, mkname(n));
    }

    for (auto &d : { 5, 10, 15, 20, 25, 30 })
    {
        double p;
        noised = noise::salt_and_pepper(im, d, &p);
        auto n = "sp_" + std::to_string(d) + "_" + std::to_string(p);
        boost::replace_all(n, ".", "_");
        process_noise(im, noised, mkname(n));
    }
}

int main()
try
{
    // setup
    std::sort(levels.begin(), levels.end());

    // process
    for (auto &n : { "lena", "cameraman.tif", "peppers" }) // cameraman, cabbages
        process_image(n);

    return 0;
}
catch (const std::exception &e)
{
    std::cerr << e.what();
    return 1;
}
catch (...)
{
    std::cerr << "Unhandled unknown exception\n";
    return 1;
}
