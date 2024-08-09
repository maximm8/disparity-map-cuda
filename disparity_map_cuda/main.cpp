#include <windows.h>

#include <chrono>

#include <opencv2/opencv.hpp> 

#include "disparity.h"

Disparity* dsp;
StereoParams params;

void gotoxy(int x, int y) 
{
    COORD pos = { x, y };
    HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleCursorPosition(output, pos);
}

void update_w(int slider, void* data)
{
    params.window_width = slider;
    dsp->SetParams(params);
}

void update_h(int slider, void* data)
{
    params.window_height = slider;
    dsp->SetParams(params);
}

void update_d1(int slider, void* data)
{    
    params.disparity_min = slider;
    dsp->SetParams(params);
}

void update_d2(int slider, void* data)
{
    params.disparity_max = slider;
    dsp->SetParams(params);
}

int main()
{
    cv::Mat img1 = cv::imread("data/left.png");
    cv::Mat img2 = cv::imread("data/right.png");

    int img_width       = img1.size().width;
    int img_height      = img1.size().height;
    int img_channels    = img1.channels();

    cv::Mat disp;

    int wind_width  = 5;
    int wind_height = 5;
    int disp_min    = 0;
    int disp_max    = 60;

    params = { wind_width, wind_height, disp_min, disp_max, CostFunction::SSD, CostOptimizer::COSTMIN };

    
    dsp = new Disparity(params, img_width, img_height, img_channels);

    // Create slider to change some parameters in relatimer 
    std::string window_params = "params";    
    cv::namedWindow(window_params, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("width", window_params, NULL, 20, update_w); cv::setTrackbarPos("width", window_params, wind_width);
    cv::createTrackbar("height", window_params, NULL, 20, update_h); cv::setTrackbarPos("height", window_params, wind_height);
    cv::createTrackbar("disp min", window_params, NULL, 100, update_d1); cv::setTrackbarPos("disp min", window_params, disp_min);
    cv::createTrackbar("disp max", window_params, NULL, 100, update_d2); cv::setTrackbarPos("disp max", window_params, disp_max);
    cv::resizeWindow(window_params, img_width, 1);


    while (true)
    {
        auto t1 = std::chrono::steady_clock::now();
        dsp->Match(img1, img2, disp);
        auto t2 = std::chrono::steady_clock::now();
        gotoxy(0, 0);
        std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;

        cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(disp, disp, cv::COLORMAP_JET);

        cv::imshow("left", img1); 
        cv::imshow("right", img2); 
        cv::imshow("disparity", disp);        

        if (cv::waitKey(1) >= 0)
            break;
    }

    return 0;
}