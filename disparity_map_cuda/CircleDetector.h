#pragma once

// std
#include <vector>
//#include <iostream>

//opencv
#include <opencv2\opencv.hpp>

struct DetectionParams
{
	////image preprocessing
	//float Scale = 1;
	//float ContrastK1 = 0;
	//float ContrastK2 = 1;

	// feature detection params
	int DiffThreshold = 20;//45
	int CountThreshold = 0; // mask5 - 9, mask7 - 12, mask9 - 15, mask11 - 21
	int CircleRadius;
	bool CleanResponse;
	bool CalcMean;

	// feature properties
	int MassMin;
	int AreaMin;
	double DensityMin;
	bool EightConnected;

	int WidthMax;
	int HeightMax;
	int XMin;
	int XMax;
	int YMin;
	int YMax;

	//bool WhightOnBlack;
	/*DetectionParams(const DetectionParams & params)
	{

	}*/
};

class CircleDetector
{
public:
	//CircleDetector(int circle_diameter, int diff_threshold, int count_threshold);
	CircleDetector(const DetectionParams& params);
	~CircleDetector();

	virtual void Detect(const cv::Mat& image);
	virtual void UpdateMean();
	virtual void UpdateDiff();

	void GetCenters(std::vector<Feature>& features);
	void GetCentersFromMean(std::vector<Feature>& features);
	void GetCentersFromDiff(std::vector<Feature>& features);

	/*void RemoveNoise(cv::Mat &circle_response_clean,
					 const DetectionParams &params);*/

	cv::Mat GetResponse() const { return CircleResponse; }

	const DetectionParams& GetParams() const { return Params; }

protected:
	DetectionParams Params;

	std::vector<std::pair<int, int>> Coordinates;

	cv::Mat Mask;
	cv::Mat CircleResponse;
	cv::Mat CircleResponseMean;
	cv::Mat CircleResponseDiff;
	int DetectionIndex = 0;
	int MeanImgsNb = 0;

	uint16_t* CircleResponsePtr;
	uint16_t* CircleResponseMeanPtr;
	uint16_t* CircleResponseDiffPtr;
	uint8_t* ImagePtr;
	uint8_t* MaskPtr;

	int64_t H, W;// h, w;// px, py;
	int64_t Mask_H, Mask_W, Mask_H2, Mask_W2;

	virtual void DetectCircleAt(int x, int y);
	void InitMask();
	void CalcCountThreshold();

	void GetCenters(uint16_t* resp_data,
		std::vector<Feature>& tracking_points,
		const DetectionParams& params);
};