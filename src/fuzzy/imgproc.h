#pragma once

#include "common.h"

#include <opencv2/core.hpp>
#include <primitives/filesystem.h>

cv::Mat imread_(path fn1, const path &fn2, bool color = false);
void imwrite_(const path &fn, const cv::Mat &m);
bool imexists_(const path &fn);
path prepare_im_fn(const path &dir, path fn);
cv::Mat float_to_uint8(cv::Mat m);

inline path prepare_im_fn(path fn)
{
    return prepare_im_fn(getOutputDir(), fn);
}

inline cv::Mat imread(path fn, bool color = false)
{
    return imread_(fn, prepare_im_fn(fn), color);
}

inline void imwrite(const path &fn, const cv::Mat &m)
{
    imwrite_(prepare_im_fn(fn), m);
}

inline bool imexists(const path &fn)
{
    return imexists_(prepare_im_fn(fn));
}

void adjustContrast(cv::Mat &frame);

/**  @function Erosion  */
void erosion(cv::Mat m, int erosion_elem = 2, int erosion_size = 0);

/** @function Dilation */
void dilation(cv::Mat m, int dilation_elem = 2, int dilation_size = 0);

// quality-metric
namespace qm
{

// sigma on block_size
double sigma(const cv::Mat &m, int i, int j, int block_size);

// Covariance
double cov(const cv::Mat &m1, const cv::Mat &m2, int i, int j, int block_size);

// Mean squared error
double eqm(const cv::Mat &img1, const cv::Mat &img2);

/**
* Compute the PSNR between 2 images
*/
double psnr(const cv::Mat &img_src, const cv::Mat &img_compressed);

/**
* Compute the SSIM between 2 images
*/
double ssim(const cv::Mat &img_src, const cv::Mat &img_compressed, int block_size, bool show_progress = false);

}

namespace noise
{

cv::Mat gauss(cv::Mat &in, double mean = 0.0, double stddev = 1.0);
cv::Mat gauss_var(cv::Mat &in, double mean = 0.0, double var = 1.0);

cv::Mat salt_and_pepper(cv::Mat &in, int d = 15, double *p = nullptr);

}
