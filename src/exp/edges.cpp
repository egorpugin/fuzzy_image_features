#include <fuzzy.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc.hpp>
//#include <range/v3/all.hpp>
#include <Eigen/Dense>

#include <primitives/date_time.h>

#include <functional>
#include <iostream>

//
auto fuzzy_pixel_1d(f64 v)
{
    auto sz = 60.0;
    auto a = -sz + v;
    auto c = v;
    auto b = sz / 2.0;

    auto x = seq(-sz + v, sz + v);
    auto f = [b, c](auto x) { return mf::pi(x, b, c); };
    auto y = x;
    std::transform(y.begin(), y.end(), y.begin(), [&f](auto e) { return f(e); });

    return y;
}

cv::Mat sobel(const cv::Mat &I, const cv::Mat &Iorig)
{
    Eigen::ArrayXd vert(6);
    vert << -1, -2, -1, 1, 2, 1;

    Eigen::ArrayXd hor(6);
    hor << 1, 2, 1, -1, -2, -1;

    Eigen::ArrayXd y(6);

    cv::Mat Igrad = cv::Mat::zeros(I.size(), I.type());

    auto mf_sz = 60;
    auto b = mf_sz / 2;

    auto sz = I.size();
    for (int i = 1; i < sz.height - 1; i++)
    {
        for (int j = 1; j < sz.width - 1; j++)
        {
#define Y_MF(xi, yj) mf::pi((f64)PIXEL_F(Iorig, i + xi, j + yj),(f64)b,(f64)PIXEL_F(Iorig, i + xi, j + yj))
#define Y_ORIG(xi, yj) (PIXEL_F(I, i + xi, j + yj))

            // x
            y <<
#define Y(xi, yj) Y_MF(xi, yj)
                Y(-1, -1),
                Y(-1, +0),
                Y(-1, +1),
                Y(+1, -1),
                Y(+1, +0),
                Y(+1, -1);
#undef Y
            y *= vert;
            auto Lfx = y.sum() / 6.0;

            y <<
#define Y(xi, yj) Y_ORIG(xi, yj)
                Y(-1, -1),
                Y(-1, +0),
                Y(-1, +1),
                Y(+1, -1),
                Y(+1, +0),
                Y(+1, -1);
#undef Y
            y *= vert;
            auto Lix = y.sum() / 6.0;

            // y
            y <<
#define Y(xi, yj) Y_MF(yj, xi)
                Y(-1, -1),
                Y(-1, +0),
                Y(-1, +1),
                Y(+1, -1),
                Y(+1, +0),
                Y(+1, -1);
#undef Y
            y *= hor;
            auto Lfy = y.sum() / 6.0;

            y <<
#define Y(xi, yj) Y_ORIG(yj, xi)
                Y(-1, -1),
                Y(-1, +0),
                Y(-1, +1),
                Y(+1, -1),
                Y(+1, +0),
                Y(+1, -1);
#undef Y
            y *= hor;
            auto Liy = y.sum() / 6.0;

            auto Lf = 1 - sqrt(Lfx * Lfx + Lfy * Lfy);
            auto Li = sqrt(Lix * Lix + Liy * Liy);
            PIXEL_F(Igrad, i, j) = (float)(Lf * Li);
        }
    }
    return Igrad;
}

/**
* Perform one thinning iteration.
* Normally you wouldn't call this function directly from your code.
*
* Parameters:
* 		im    Binary image with range = [0,1]
* 		iter  0=even, 1=odd
*/
void thinningIteration(cv::Mat& img, int iter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;    // north (pAbove)
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;    // south (pBelow)

    uchar *pDst;

    // initialize row pointers
    pAbove = NULL;
    pCurr = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    for (y = 1; y < img.rows - 1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr = pBelow;
        pBelow = img.ptr<uchar>(y + 1);

        pDst = marker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (x = 1; x < img.cols - 1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x + 1]);
            we = me;
            me = ea;
            ea = &(pCurr[x + 1]);
            sw = so;
            so = se;
            se = &(pBelow[x + 1]);

            int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
        }
    }

    img &= ~marker;
}

/**
* Function for thinning the given binary image
*
* Parameters:
* 		src  The source image, binary with range = [0,255]
* 		dst  The destination image
*/
void thinning(const cv::Mat& src, cv::Mat& dst)
{
    dst = src.clone();
    dst /= 255;         // convert to binary image

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

// threashold specifying minimum area of a blob
void remove_small_blobs(cv::Mat &input, double threshold)
{
    using namespace cv;
    using namespace std;
    // your input binary image
    // assuming that blob pixels have positive values, zero otherwise


    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<int> small_blobs;
    double contour_area;
    Mat temp_image;

    // find all contours in the binary image
    input.copyTo(temp_image);
    findContours(temp_image, contours, hierarchy, CV_RETR_CCOMP,
        CV_CHAIN_APPROX_SIMPLE);

    // Find indices of contours whose area is less than `threshold`
    if (!contours.empty()) {
        for (size_t i = 0; i < contours.size(); ++i) {
            contour_area = contourArea(contours[i]);
            if (contour_area < threshold)
                small_blobs.push_back(i);
        }
    }

    // fill-in all small contours with zeros
    for (size_t i = 0; i < small_blobs.size(); ++i) {
        drawContours(input, contours, small_blobs[i], cv::Scalar(0),
            CV_FILLED, 8);
    }
}

// fill holes
cv::Mat imfill(cv::Mat src)
{
    // Floodfill from point (0, 0)
    cv::Mat im_floodfill = src.clone();
    cv::floodFill(im_floodfill, cv::Point(0, 0), cv::Scalar(255));

    // Invert floodfilled image
    cv::Mat im_floodfill_inv;
    cv::bitwise_not(im_floodfill, im_floodfill_inv);

    // Combine the two images to get the foreground.
    return src | im_floodfill_inv;
}

std::vector<cv::Point> contoursConvexHull(const std::vector<std::vector<cv::Point>> &contours)
{
    std::vector<cv::Point> result;
    std::vector<cv::Point> pts;
    for (size_t i = 0; i < contours.size(); i++)
        for (size_t j = 0; j < contours[i].size(); j++)
            pts.push_back(contours[i][j]);
    cv::convexHull(pts, result);
    return result;
}

void close(cv::Mat m)
{
    std::vector<std::vector<cv::Point> > contours;
    findContours(m, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> ConvexHullPoints = contoursConvexHull(contours);

    cv::Mat drawing = cv::Mat::zeros(m.size(), CV_8UC3);

    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(255, 255, 255);
        cv::drawContours(drawing, contours, i, color, 2);
    }

    polylines(drawing, ConvexHullPoints, true, cv::Scalar(0, 0, 255), 2);
    imshow("Contours", drawing);

    polylines(m, ConvexHullPoints, true, cv::Scalar(0, 0, 255), 2);
    imshow("contoursConvexHull", m);

}

int main(int argc, char **argv)
{
    auto Ig = cv::imread("01_dr2.png", 0);
    if (Ig.empty())
        throw std::runtime_error("image not found");

    auto mask = cv::imread("01_dr_mask.tif", 0);
    cv::Mat Ig2;
    Ig.copyTo(Ig2, mask);
    Ig = Ig2;

    //adjustContrast(Ig);

    cv::Mat I;
    Ig.convertTo(I, CV_32F);

    cv::Mat I2h = cv::Mat::zeros(I.size(), I.type());
    cv::Mat I2v = cv::Mat::zeros(I.size(), I.type());
    cv::Mat I2d4 = cv::Mat::zeros(I.size(), I.type());
    cv::Mat I2d8 = cv::Mat::zeros(I.size(), I.type());
    cv::Mat I2dd = cv::Mat::zeros(I.size(), I.type());

    auto sz = I.size();
    for (int i = 1; i < sz.height - 1; i++)
    {
        for (int j = 1; j < sz.width - 1; j++)
        {
            PIXEL_F(I2h, i, j) = (PIXEL_F(I, i, j - 1) + PIXEL_F(I, i, j + 1)) / 2.0f;
            PIXEL_F(I2v, i, j) = (PIXEL_F(I, i - 1, j) + PIXEL_F(I, i + 1, j)) / 2.0f;
            PIXEL_F(I2d4, i, j) = (PIXEL_F(I, i - 1, j) + PIXEL_F(I, i + 1, j) +
                PIXEL_F(I, i, j - 1) + PIXEL_F(I, i, j + 1)) /
                4.0f;
            PIXEL_F(I2dd, i, j) = (PIXEL_F(I, i - 1, j - 1) + PIXEL_F(I, i + 1, j + 1) +
                PIXEL_F(I, i + 1, j - 1) + PIXEL_F(I, i - 1, j + 1)) /
                4.0f;
            PIXEL_F(I2d8, i, j) = (PIXEL_F(I, i - 1, j) + PIXEL_F(I, i + 1, j) +
                PIXEL_F(I, i, j - 1) + PIXEL_F(I, i, j + 1) +
                PIXEL_F(I, i - 1, j - 1) + PIXEL_F(I, i + 1, j + 1) +
                PIXEL_F(I, i + 1, j - 1) + PIXEL_F(I, i - 1, j + 1)) /
                8.0f;
        }
    }

    //fuzzy_pixel_1d(PIXEL_F(I, 467 + 70, 306 + 55));

    auto Ismall = Ig;
    //auto Ismall = cv::Mat(Ig, cv::Rect(467, 306, 626 - 467, 465 - 306));

    cv::Mat Igrad_x;
    cv::Mat Igrad_y;
    cv::Mat Igrad_x_abs;
    cv::Mat Igrad_y_abs;
    cv::Mat grad;
    cv::GaussianBlur(Ismall, grad, cv::Size(3, 3), 0);
    cv::Sobel(grad, Igrad_x, CV_64F, 1, 0);
    cv::Sobel(grad, Igrad_y, CV_64F, 0, 1);
    cv::convertScaleAbs(Igrad_x, Igrad_x_abs);
    cv::convertScaleAbs(Igrad_y, Igrad_y_abs);
    cv::addWeighted(Igrad_x_abs, 0.5, Igrad_y_abs, 0.5, 0, grad);
    grad = grad > 15;
    //thinning(grad, grad);
    //adjustContrast(grad);
    //grad = 255 - grad;
    remove_small_blobs(grad, 200);
    grad = 255 - grad;
    //thinning(grad, grad);
    //cv::medianBlur(grad, grad, 3);

    cv::Mat canny;
    cv::Canny(Ismall, canny, 150, 180);

    cv::Mat Igrad;
    /*auto us = get_time<std::chrono::milliseconds>([&]
    {
    sobel(
    cv::Mat(I2d8, cv::Rect(467, 306, 626 - 467, 465 - 306)),
    cv::Mat(I, cv::Rect(467, 306, 626 - 467, 465 - 306))).convertTo(Igrad, CV_8U);
    });
    std::cout << "4 = " << us << " ms.\n";*/
    auto us = get_time<std::chrono::milliseconds>([&]
    {
        sobel(
            I2d8,
            I).convertTo(Igrad, CV_8U);
    });
    //std::cout << "5 = " << us << " ms.\n";

    //Igrad = cv::Mat(Igrad, cv::Rect(467, 306, 626 - 467, 465 - 306));
    Igrad = Igrad > 4.5f;

    //int cut = 10;
    //Igrad = cv::Mat(Igrad, cv::Rect(cut, cut, sz.width - cut - 1, sz.height - cut - 1));

    //cv::medianBlur(Igrad, Igrad, 3);
    //adjustContrast(Igrad);
    //Igrad = 255 - Igrad;
    remove_small_blobs(Igrad, 50);
    cv::imwrite("our_not_thin.png", Igrad);
    //cv::medianBlur(Igrad, Igrad, 5);
    //Dilation(Igrad);
    //Dilation(Igrad);
    //Erosion(Igrad);
    thinning(Igrad, Igrad);
    cv::imwrite("our_thin.png", Igrad);
    //cv::medianBlur(Igrad, Igrad, 3);
    //Igrad = imfill(Igrad);

    //cv::imshow("test4", Igrad);
    //cv::imwrite("our1.png", Igrad);

    //Igrad = cv::imread("our1.png", 0);
    //close(Igrad);


    cv::imwrite("sobel.png", grad);
    //cv::imwrite("canny.png", canny);
    cv::imwrite("our.png", Igrad);

    //cv::imshow("test", Ismall);
    cv::imshow("test2", grad);
    //cv::imshow("test3", canny);
    cv::imshow("test4", Igrad);
    cv::waitKey();

    return 0;
}
