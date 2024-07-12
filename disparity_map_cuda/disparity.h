#pragma once

#include <opencv2/opencv.hpp> 

//#include <cuda.h>
#include <cuda_runtime.h>

enum CostFunction  { SAD, SSD, NCC, ZNCC};
enum CostOptimizer { COSTMIN, COSTMAX};

struct StereoParams
{
    int window_width;
    int window_height;
    int disparity_min;
    int disparity_max;

    CostFunction cost;
    CostOptimizer optim;
};

typedef bool (*pcost_optimiser)(float, float);
typedef float (*pcost_func)(unsigned char* left, unsigned char* right, int x, int y, int d, int w, int h, int chs, int ww, int wh);

class Disparity
{
public:
    Disparity(const StereoParams& params_, int windth, int height, int chanles);
    ~Disparity();

    void SetParams(const StereoParams &params_);
    void Match(const cv::Mat& img1, const cv::Mat& img2, cv::Mat &disparity_map);

private:

    int width;
    int height;
    int channels;

    StereoParams params;

    unsigned char*  img1_dev;
    unsigned char*  img2_dev;
    float*          disparity_dev;
    float*          cost_dev;

    //host function pointer
    pcost_func      host_cost_func_ptr;
    pcost_optimiser host_cost_optim_ptr;

};
