#include <opencv2/opencv.hpp> 

#include <librealsense2/rs.hpp> 

#include "disparity.h"

int main_()
{
    int img_width   = 640;
    int img_height  = 480;
    int fps         = 30;

    rs2::pipeline pipe;
    rs2::config cfg;

    // enable image strems
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, img_width, img_height, RS2_FORMAT_Y8, fps);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, img_width, img_height, RS2_FORMAT_Y8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, img_width, img_height, RS2_FORMAT_Z16, fps);

    pipe.start(cfg);
    
    cv::Mat disp, disp_color;
    cv::Mat depth_color;

    StereoParams params = { 11, 11, 0, 70, CostFunction::SAD, CostOptimizer::COSTMIN };
    //StereoParams params = { 11, 11, 0, 70, CostFunction::NCC, CostOptimizer::COSTMAX };

    Disparity dsp(params, img_width, img_height, 1);

    while (true)
    {
        // get a data
        rs2::frameset frames = pipe.wait_for_frames();

        //rs2::frame color_frame = frames.get_color_frame();
        rs2::frame frame1 = frames.get_infrared_frame(1);
        rs2::frame frame2 = frames.get_infrared_frame(2);
        rs2::frame depth_frame = frames.get_depth_frame();

        // create an opencv images
        //cv::Mat color(cv::Size(img_width, img_height), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat img1(cv::Size(img_width, img_height), CV_8UC1, (void*)frame1.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat img2(cv::Size(img_width, img_height), CV_8UC1, (void*)frame2.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(img_width, img_height), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        // compute disparity map
        dsp.Match(img1, img2, disp);

        // normilise disparity and depth images
        cv::normalize(disp, disp_color, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(disp_color, disp_color, cv::COLORMAP_JET);

        cv::normalize(depth, depth_color, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(depth_color, depth_color, cv::COLORMAP_JET);

        // display
        //cv::imshow("color", color);
        cv::imshow("left", img1);
        cv::imshow("right", img2);
        cv::imshow("disparity cuda", disp_color);
        cv::imshow("depth", depth_color);

        if (cv::waitKey(1) >= 0)
            break;
    }

    return 0;
}