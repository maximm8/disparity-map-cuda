#include <iostream>
#include <cstdlib>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp> 

#include "disparity.h"


void checkCudaError(cudaError_t err, const char* msg) 
{

    if (err != cudaSuccess) 
    {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

__device__  bool cost_min(float cost,  float cost_best)
{
    if (cost < cost_best)
        return true;

    return false;
}

__device__  bool cost_max(float cost, float cost_best)
{
    if (cost > cost_best)
        return true;

    return false;
}

__device__  float sad_cost(unsigned char* left, unsigned char* right, int x, int y, int d, int w, int h, int chs, int ww, int wh)
{
    int ww2 = ww / 2;
    int wh2 = wh / 2;

    float cost_ = 0;
    for (int64_t yy = 0; yy < wh; ++yy)
    {
        for (int64_t xx = 0; xx < ww; ++xx)
        {
            int16_t j1 = y + yy - ww2;
            int16_t i1 = x + xx - wh2;

            int16_t j2 = y + yy - ww2;
            int16_t i2 = (x - d) + xx - wh2;

            if ((i1 > -1 && j1 > -1 && i1 < w && j1 < h) && (i2 > -1 && j2 > -1 && i2 < w && j2 < h))
            {
                int idx1 = j1 * w + i1;
                int idx2 = j2 * w + i2;

                for (int64_t ch = 0; ch < chs; ++ch)
                    cost_ += abs(left[chs * idx1 + ch] - right[chs * idx2 + ch]);
            }

        }
    }

    return cost_;
}

__device__  float ssd_cost(unsigned char* left, unsigned char* right, int x, int y, int d, int w, int h, int chs, int ww, int wh)
{
    int ww2 = ww / 2;
    int wh2 = wh / 2;

    float cost_ = 0;
    for (int64_t yy = 0; yy < wh; ++yy)
    {
        for (int64_t xx = 0; xx < ww; ++xx)
        {
            int16_t j1 = y + yy - ww2;
            int16_t i1 = x + xx - wh2;

            int16_t j2 = y + yy - ww2;
            int16_t i2 = (x - d) + xx - wh2;

            if ((i1 > -1 && j1 > -1 && i1 < w && j1 < h) && (i2 > -1 && j2 > -1 && i2 < w && j2 < h))
            {
                int idx1 = j1 * w + i1;
                int idx2 = j2 * w + i2;

                for (int64_t ch = 0; ch < chs; ++ch)
                {
                    float t = (left[chs * idx1 + ch] - right[chs * idx2 + ch]);
                    cost_ += t * t;
                }
            }

        }
    }

    return cost_;
}

__device__  float ncc_cost(unsigned char* left, unsigned char* right, int x, int y, int d, int w, int h, int chs, int ww, int wh)
{
    int ww2 = ww / 2;
    int wh2 = wh / 2;

    float cost_ = 0;
    float lr = 0;
    float llsum = 0;
    float rrsum = 0;

    for (int64_t yy = 0; yy < wh; ++yy)
    {
        for (int64_t xx = 0; xx < ww; ++xx)
        {
            int16_t j1 = y + yy - ww2;
            int16_t i1 = x + xx - wh2;

            int16_t j2 = y + yy - ww2;
            int16_t i2 = (x - d) + xx - wh2;

            if ((i1 > -1 && j1 > -1 && i1 < w && j1 < h) && (i2 > -1 && j2 > -1 && i2 < w && j2 < h))
            {
                int idx1 = j1 * w + i1;
                int idx2 = j2 * w + i2;

                //int ch = 0;
                for (int64_t ch = 0; ch < chs; ++ch)
                {
                    lr += ((float)left[chs * idx1 + ch] * (float)right[chs * idx2 + ch]);
                    llsum += ((float)left[chs * idx1 + ch] * (float)left[chs * idx1 + ch]);
                    rrsum += ((float)right[chs * idx2 + ch] * (float)right[chs * idx2 + ch]);
                }
            }
        }
    }

    cost_ = lr / sqrt(llsum * rrsum);

    return cost_;
}

__device__  void mean(unsigned char* left, unsigned char* right, int x, int y, int d, int w, int h, int chs, int ww, int wh, float &lm, float &rm)
{
    int ww2 = ww / 2;
    int wh2 = wh / 2;

    lm = 0;
    rm = 0;
    for (int64_t yy = 0; yy < wh; ++yy)
    {
        for (int64_t xx = 0; xx < ww; ++xx)
        {
            int16_t j1 = y + yy - ww2;
            int16_t i1 = x + xx - wh2;

            int16_t j2 = y + yy - ww2;
            int16_t i2 = (x - d) + xx - wh2;

            if ((i1 > -1 && j1 > -1 && i1 < w && j1 < h) && (i2 > -1 && j2 > -1 && i2 < w && j2 < h))
            {
                int idx1 = j1 * w + i1;
                int idx2 = j2 * w + i2;

                //int ch = 0;
                for (int64_t ch = 0; ch < chs; ++ch)
                {
                    lm += left[chs * idx1 + ch];
                    rm += right[chs * idx2 + ch];
                }
            }
        }
    }

    float N = ww * wh;
    lm /= N;
    rm /= N;

}


__device__  float zncc_cost(unsigned char* left, unsigned char* right, int x, int y, int d, int w, int h, int chs, int ww, int wh)
{
    int ww2 = ww / 2;
    int wh2 = wh / 2;

    float cost_ = 0;
    float lr = 0;
    float llsum = 0;
    float rrsum = 0;

    float lm = 0;
    float rm = 0;

    mean(left, right, x, y, d, w, h, chs, ww, wh, lm, rm);

    for (int64_t yy = 0; yy < wh; ++yy)
    {
        for (int64_t xx = 0; xx < ww; ++xx)
        {
            int16_t j1 = y + yy - ww2;
            int16_t i1 = x + xx - wh2;

            int16_t j2 = y + yy - ww2;
            int16_t i2 = (x - d) + xx - wh2;

            if ((i1 > -1 && j1 > -1 && i1 < w && j1 < h) && (i2 > -1 && j2 > -1 && i2 < w && j2 < h))
            {
                int idx1 = j1 * w + i1;
                int idx2 = j2 * w + i2;

                //int ch = 0;
                for (int64_t ch = 0; ch < chs; ++ch)
                {
                    lr += ((float)left[chs * idx1 + ch]-lm) *  ((float)right[chs * idx2 + ch]-rm);
                    llsum += ( ((float)left[chs * idx1 + ch] - lm) *((float)left[chs * idx1 + ch]-lm) );
                    rrsum += ( ((float)right[chs * idx2 + ch]-rm) * ((float)right[chs * idx2 + ch] -rm) );
                }
            }
        }
    }

    cost_ = lr / sqrt(llsum * rrsum);

    return cost_;
}


__device__ pcost_optimiser cost_min_ptr = cost_min;
__device__ pcost_optimiser cost_max_ptr = cost_max;

__device__ pcost_func sad_cost_ptr      = sad_cost;
__device__ pcost_func ssd_cost_ptr      = ssd_cost;
__device__ pcost_func ncc_cost_ptr      = ncc_cost;
__device__ pcost_func zncc_cost_ptr     = zncc_cost;

__global__ void disparity_kernel(unsigned char* left, unsigned char* right, float* out_disp, float* out_cost, int w, int h, int ch, int ww, int wh, int d1, int d2,  pcost_func calc_cost, pcost_optimiser optim_cost)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int ww2 = ww / 2;
    int wh2 = wh / 2;

    if (x - ww2 - d2 < 0 || x + ww2 + 1>w)
    {
        out_disp[(y + wh2) * w + x + ww2] = 0;
        out_cost[(y + wh2) * w + x + ww2] = 0;
        return;
    }

    //float cost[100];
    float cost_best = 1000000;
    int16_t disp_ind = -1000000;
    bool init = false;

    for (int64_t d = d1; d < d2; ++d)
    {
        float cost_ = calc_cost(left, right, x, y, d, w, h, ch, ww, wh);
        //float cost_ = sad_cost(left, right, x, y, d, w, h, ch, ww, wh);

        if (!init || optim_cost(cost_, cost_best))
        //if (disp_ind == -1000000 || cost_min(cost_, cost_best))
        {
            cost_best = cost_;
            disp_ind = d - d1;
            init = true;
        }
    }

    out_disp[(y + wh2) * w + x + ww2] = disp_ind;
    out_cost[(y + wh2) * w + x + ww2] = cost_best;
}

Disparity::Disparity(const StereoParams& params_, int width_, int height_, int channles_)
{
    width = width_;
    height = height_;
    channels = channles_;

    checkCudaError(cudaMalloc((void**)&(img1_dev), width * height * channels * sizeof(unsigned char)), "cudaMalloc img_dev1");
    checkCudaError(cudaMalloc((void**)&(img2_dev), width * height * channels * sizeof(unsigned char)), "cudaMalloc img_dev2");
    checkCudaError(cudaMalloc((void**)&(disparity_dev), width * height * sizeof(float)), "cudaMalloc dispairy_dev");
    checkCudaError(cudaMalloc((void**)&(cost_dev), width * height * sizeof(float)), "cudaMalloc cost_dev");

    SetParams(params_);
}

void Disparity::SetParams(const StereoParams& params_)
{
    params = params_;
    
    switch (params.cost)
    {
        case CostFunction::SAD:
            cudaMemcpyFromSymbol(&host_cost_func_ptr, sad_cost_ptr, sizeof(pcost_func));
            break;
        case CostFunction::SSD:
            cudaMemcpyFromSymbol(&host_cost_func_ptr, ssd_cost_ptr, sizeof(pcost_func));
            break;
        case CostFunction::NCC:
            cudaMemcpyFromSymbol(&host_cost_func_ptr, ncc_cost_ptr, sizeof(pcost_func));
            break;
        case CostFunction::ZNCC:
            cudaMemcpyFromSymbol(&host_cost_func_ptr, zncc_cost_ptr, sizeof(pcost_func));
            break;
        default:
            cudaMemcpyFromSymbol(&host_cost_func_ptr, sad_cost_ptr, sizeof(pcost_func));
    }

    switch (params.optim)
    {
        case CostOptimizer::COSTMIN:
            cudaMemcpyFromSymbol(&host_cost_optim_ptr, cost_min_ptr, sizeof(pcost_optimiser));
            break;
        case CostOptimizer::COSTMAX:
            cudaMemcpyFromSymbol(&host_cost_optim_ptr, cost_max_ptr, sizeof(pcost_optimiser));
            break;
        default:
            cudaMemcpyFromSymbol(&host_cost_optim_ptr, cost_min_ptr, sizeof(pcost_optimiser));
            break;
    }
}

void Disparity::Match(const cv::Mat& img1, const cv::Mat& img2, cv::Mat &disparity_map)// int ww, int wh, int d1, int d2)
{
    disparity_map = cv::Mat(img1.rows, img1.cols, CV_32FC1);
    int ww = params.window_width;
    int wh = params.window_height;
    int d1 = params.disparity_min;
    int d2 = params.disparity_max;
    
    checkCudaError(cudaMemcpy(img1_dev, img1.data, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy img1_dev");
    checkCudaError(cudaMemcpy(img2_dev, img2.data, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy img2_dev");

    dim3 blockDims(16, 16);
    dim3 gridDims((width + blockDims.x - 1) / blockDims.x, (height + blockDims.y - 1) / blockDims.y);
    
    disparity_kernel <<< gridDims, blockDims >>>(img1_dev, img2_dev, disparity_dev, cost_dev, width, height, channels, ww, wh, d1, d2, host_cost_func_ptr, host_cost_optim_ptr);

    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    checkCudaError(cudaMemcpy(disparity_map.data, disparity_dev, width * height * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy disparity_map");

#if DEBUG
    cv::Mat output_cost = cv::Mat(img1.rows, img1.cols, CV_32FC1);    
    checkCudaError(cudaMemcpy(output_cost.data, cost_dev, width * height * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy disparity_map cost");
#endif
}


Disparity::~Disparity()
{
    cudaFree(img1_dev);
    cudaFree(img2_dev);
    cudaFree(disparity_dev);
    cudaFree(cost_dev);
}