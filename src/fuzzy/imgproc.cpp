#include "imgproc.h"

#include <opencv2/core/core_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void adjustContrast(cv::Mat &frame)
{
    static auto clahe = cv::createCLAHE();
    clahe->apply(frame, frame);
}

/**  @function Erosion  */
void erosion(cv::Mat m, int erosion_elem, int erosion_size)
{
    int erosion_type;
    if (erosion_elem == 0) { erosion_type = cv::MORPH_RECT; }
    else if (erosion_elem == 1) { erosion_type = cv::MORPH_CROSS; }
    else if (erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }

    cv::Mat element = cv::getStructuringElement(erosion_type,
        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        cv::Point(erosion_size, erosion_size));

    /// Apply the erosion operation
    cv::erode(m, m, element);
}

/** @function Dilation */
void dilation(cv::Mat m, int dilation_elem, int dilation_size)
{
    int dilation_type;
    if (dilation_elem == 0) { dilation_type = cv::MORPH_RECT; }
    else if (dilation_elem == 1) { dilation_type = cv::MORPH_CROSS; }
    else if (dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }

    cv::Mat element = cv::getStructuringElement(dilation_type,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    cv::dilate(m, m, element);
}

path prepare_im_fn(const path &dir, path fn)
{
    if (fs::exists(fn))
        return fn;
    if (fn.is_absolute())
        return fn;
    if (!fn.has_extension())
        fn += ".png";
    if (fn.empty() || *fn.begin() != dir)
        fn = dir / fn;
    return fn;
}

cv::Mat imread_(path fn1, const path &fn2, bool color)
{
    const auto err = "Cannot read image: " + fn1.string();

    auto read = [&color, &err](const auto &fn)
    {
        auto m = cv::imread(fn.string(), color ? 1 : 0);
        if (m.empty())
            throw std::runtime_error(err);
        return m;
    };

    auto source_fn = fn1;
    if (!source_fn.is_absolute())
        source_fn = path("d:/Institute/Postgraduate/experimental/images") / source_fn;
    if (!source_fn.has_extension())
        source_fn += ".png";
    if (fs::exists(source_fn))
        return read(source_fn);
    if (imexists(fn2))
        return read(fn2);
    throw std::runtime_error(err);
}

void imwrite_(const path &fn, const cv::Mat &m)
{
    fs::create_directories(fn.parent_path());
    cv::imwrite(fn.string(), m);
}

bool imexists_(const path &fn)
{
    return fs::exists(fn);
}

// quality-metric
namespace qm
{

#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 255)

// sigma on block_size
double sigma(const cv::Mat &m, int i, int j, int block_size)
{
    double sd = 0;

    cv::Mat m_tmp = m(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
    cv::Mat m_squared(block_size, block_size, CV_64F);

    cv::multiply(m_tmp, m_tmp, m_squared);

    // E(x)
    double avg = mean(m_tmp)[0];
    // E(x²)
    double avg_2 = mean(m_squared)[0];


    sd = sqrt(avg_2 - avg * avg);

    return sd;
}

// Covariance
double cov(const cv::Mat &m1, const cv::Mat &m2, int i, int j, int block_size)
{
    cv::Mat m3 = cv::Mat::zeros(block_size, block_size, m1.depth());
    cv::Mat m1_tmp = m1(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
    cv::Mat m2_tmp = m2(cv::Range(i, i + block_size), cv::Range(j, j + block_size));

    cv::multiply(m1_tmp, m2_tmp, m3);

    double avg_ro = cv::mean(m3)[0]; // E(XY)
    double avg_r = cv::mean(m1_tmp)[0]; // E(X)
    double avg_o = cv::mean(m2_tmp)[0]; // E(Y)

    double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)

    return sd_ro;
}

// Mean squared error
double eqm(const cv::Mat &img1, const cv::Mat &img2)
{
    int i, j;
    double eqm = 0;
    int height = img1.rows;
    int width = img1.cols;

    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
            eqm += (img1.at<uint8_t>(i, j) - img2.at<uint8_t>(i, j)) * (img1.at<uint8_t>(i, j) - img2.at<uint8_t>(i, j));

    eqm /= height * width;

    return eqm;
}

/**
*	Compute the PSNR between 2 images
*/
double psnr(const cv::Mat &img_src, const cv::Mat &img_compressed)
{
    int D = 255;
    return (10 * log10((D*D) / eqm(img_src, img_compressed)));
}

/**
* Compute the SSIM between 2 images
*/
double ssim(const cv::Mat &img_src, const cv::Mat &img_compressed, int block_size, bool show_progress)
{
    double ssim = 0;

    int nbBlockPerHeight = img_src.rows / block_size;
    int nbBlockPerWidth = img_src.cols / block_size;

    for (int k = 0; k < nbBlockPerHeight; k++)
    {
        for (int l = 0; l < nbBlockPerWidth; l++)
        {
            int m = k * block_size;
            int n = l * block_size;

            double avg_o = mean(img_src(cv::Range(k, k + block_size), cv::Range(l, l + block_size)))[0];
            double avg_r = mean(img_compressed(cv::Range(k, k + block_size), cv::Range(l, l + block_size)))[0];
            double sigma_o = sigma(img_src, m, n, block_size);
            double sigma_r = sigma(img_compressed, m, n, block_size);
            double sigma_ro = cov(img_src, img_compressed, m, n, block_size);

            ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));
        }
        // Progress
        //if (show_progress)
            // std::cout << "\r>>SSIM [" << (int)((((double)k) / nbBlockPerHeight) * 100) << "%]";
    }
    ssim /= nbBlockPerHeight * nbBlockPerWidth;

    if (show_progress)
    {
        //std::cout << "\r>>SSIM [100%]" << std::endl;
        //std::cout << "SSIM : " << ssim << std::endl;
    }

    return ssim;
}

}

namespace noise
{

cv::Mat gauss(cv::Mat &in, double mean, double stddev)
{
    auto noise = cv::Mat(in.size(), CV_64F);
    cv::Mat res;
    cv::normalize(in, res, 0.0, 1.0, CV_MINMAX, CV_64F);
    cv::randn(noise, mean, stddev);
    res = res + noise;
    //cv::normalize(res, res, 0.0, 1.0, CV_MINMAX, CV_64F);
    res.convertTo(res, CV_8UC1, 255, 0);
    return res;
}

cv::Mat gauss_var(cv::Mat &in, double mean, double var)
{
    return gauss(in, mean, sqrt(var));
}

cv::Mat salt_and_pepper(cv::Mat &in, int d, double *p)
{
    cv::Mat saltpepper_noise = cv::Mat::zeros(in.rows, in.cols, CV_8U);
    randu(saltpepper_noise, 0, 255);

    cv::Mat black = saltpepper_noise < d;
    cv::Mat white = saltpepper_noise > (255 - d);

    if (p)
        *p = d * 2 / 256.0;

    cv::Mat saltpepper_img = in.clone();
    saltpepper_img.setTo(255, white);
    saltpepper_img.setTo(0, black);
    return saltpepper_img;
}

}

cv::Mat float_to_uint8(cv::Mat in)
{
    double min, max;
    cv::minMaxLoc(in, &min, &max);
    in = in / max * 255.0;
    cv::Mat out;
    in.convertTo(out, CV_8U);
    return out;
}
