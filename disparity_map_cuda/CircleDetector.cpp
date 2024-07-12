#include "CircleDetector.h"

CircleDetector::CircleDetector(const DetectionParams& params): Params(params)
{
	InitMask();
	CalcCountThreshold();
}

CircleDetector::CircleDetector(const DetectionParams& params)
	: Params(params)
{
	InitMask();
	CalcCountThreshold();
}


CircleDetector::~CircleDetector()
{
}

void CircleDetector::InitMask()
{
	int circle_diameter = Params.CircleRadius * 2 + 1;
	if (Params.CircleRadius <= 2)
	{
		unsigned char* mask = new unsigned char[circle_diameter * circle_diameter]
		{
			0, 1, 1, 1, 0,
				1, 0, 0, 0, 1,
				1, 0, 0, 0, 1,
				1, 0, 0, 0, 1,
				0, 1, 1, 1, 0,
		};
		Mask = cv::Mat(circle_diameter, circle_diameter, CV_8UC1, mask);
	}
	else if (Params.CircleRadius == 3)
	{
		unsigned char* mask = new unsigned char[circle_diameter * circle_diameter]
		{
			0, 0, 1, 1, 1, 0, 0,
				0, 1, 0, 0, 0, 1, 0,
				1, 0, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 0, 1,
				0, 1, 0, 0, 0, 1, 0,
				0, 0, 1, 1, 1, 0, 0,
		};
		Mask = cv::Mat(circle_diameter, circle_diameter, CV_8UC1, mask);
	}
	else if (Params.CircleRadius == 4)
	{
		unsigned char* mask = new unsigned char[circle_diameter * circle_diameter]
		{
			0, 0, 0, 1, 1, 1, 0, 0, 0,
				0, 0, 1, 0, 0, 0, 1, 0, 0,
				0, 1, 0, 0, 0, 0, 0, 1, 0,
				1, 0, 0, 0, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 0, 0, 0, 1,
				0, 1, 0, 0, 0, 0, 0, 1, 0,
				0, 0, 1, 0, 0, 0, 1, 0, 0,
				0, 0, 0, 1, 1, 1, 0, 0, 0,
		};
		Mask = cv::Mat(circle_diameter, circle_diameter, CV_8UC1, mask);
	}

	else if (Params.CircleRadius == 5)
	{

		unsigned char* mask = new unsigned char[circle_diameter * circle_diameter]
		{
			0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
				0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
				0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
				1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
				0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
				0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0
		};

		Mask = cv::Mat(circle_diameter, circle_diameter, CV_8UC1, mask);
	}
	else
	{
		//custom size
		Mask = cv::Mat::zeros(circle_diameter, circle_diameter, CV_8UC1);

		cv::circle(Mask, cv::Point(Params.CircleRadius, Params.CircleRadius), Params.CircleRadius + 1, cv::Scalar(1, 1, 1), 1);
		Mask.at<uint8_t>(Params.CircleRadius, 0) = 1;
		Mask.at<uint8_t>(0, Params.CircleRadius) = 1;
		Mask.at<uint8_t>(Params.CircleRadius, Mask.size().width - 1) = 1;
		Mask.at<uint8_t>(Mask.size().height - 1, Params.CircleRadius) = 1;
	}

	Mask_H = Mask.size().height;
	Mask_W = Mask.size().width;
	Mask_H2 = (Mask_H - 1) / 2;
	Mask_W2 = (Mask_W - 1) / 2;

	MaskPtr = Mask.data;
}

void CircleDetector::CalcCountThreshold()
{
	if (Params.CountThreshold == 0)
	{
		int count = 0;
		for (int i = 0; i < Mask.size().area(); ++i)
			if (MaskPtr[i] == 1)
			{
				count++;
			}

		Params.CountThreshold = count * 0.75;
	}
}


void CircleDetector::Detect(const cv::Mat& image)
{
	DetectionIndex++;

	ImagePtr = image.data;

	CircleResponse = cv::Mat::zeros(image.size(), CV_16UC1);
	CircleResponsePtr = (uint16_t*)CircleResponse.data;

	H = image.size().height;
	W = image.size().width;

	int y_lim = MIN(H, Params.YMax);
	int x_lim = MIN(W, Params.XMax);

#pragma omp parallel for
	for (int64_t y = Params.YMin + Mask_H2; y < y_lim - Mask_H2; ++y)
	{
#pragma omp parallel for
		for (int64_t x = Params.XMin + Mask_W2; x < x_lim - Mask_W2; ++x)
		{
			DetectCircleAt(x, y);
		}
	}

	if (Params.CleanResponse)
	{
		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		cv::erode(CircleResponse, CircleResponse, element);
	}

	if (Params.CalcMean && CircleResponseMean.size().width == 0)
	{
		CircleResponseMean = cv::Mat::zeros(CircleResponse.size(), CircleResponse.type());
		CircleResponseMeanPtr = (uint16_t*)CircleResponseMean.data;
	}
}

void CircleDetector::UpdateDiff()
{
	if (!Params.CalcMean)
		return;

	cv::absdiff(CircleResponse, CircleResponseMean, CircleResponseDiff);
	CircleResponseDiffPtr = (uint16_t*)CircleResponseDiff.data;
}


void CircleDetector::UpdateMean()
{
	if (!Params.CalcMean)
		return;

	/*DetectionParams p;
	p.DiffThreshold = 0;
	p.CountThreshold = 0;
	p.MassMin = 0;
	p.AreaMin = 3;
	p.DensityMin = 0;
	p.EightConnected = false;
	p.WidthMax = 10000;
	p.HeightMax = 10000;
	p.XMin = 0;
	p.XMax = 10000;
	p.YMin = 0;
	p.YMax = 10000;
*/
	MeanImgsNb++;

	//cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));	
	//cv::erode(CircleResponse, CircleResponse, element);

	//cv::Mat circle_response_clean = CircleResponse.clone();
	cv::Mat circle_response_clean = CircleResponse;

	//cv::Mat circle_response_clean;
	//RemoveNoise(circle_response_clean, p);

	//CircleResponseMean = CircleResponseMean*0.5f + CircleResponse*0.5f;
	//CircleResponseMean = CircleResponseMean*0.5f + circle_response_clean*0.5f;


	CircleResponseMean = (CircleResponseMean * (MeanImgsNb - 1) + circle_response_clean) / MeanImgsNb;

	/*uint16_t *resp_data1 = reinterpret_cast<uint16_t*> (CircleResponseMean.data);
	uint16_t *resp_data2 = reinterpret_cast<uint16_t*> (circle_response_clean.data);

	int w = W;
	int h = H;

	for (size_t offset = 0; offset < W*H; ++offset)
	{
		if (resp_data1[offset] > 0 && resp_data2[offset] > 0)
			resp_data1[offset] = resp_data1[offset] * 0.5 + resp_data2[offset] * 0.5;
		else if (resp_data2[offset] > 0)
			resp_data1[offset] = resp_data2[offset];
	}*/
}


//void CircleDetector::DetectMT(const cv::Mat &image, int threads_nb)
//{
//	//Lauch parts-1 threads
//	for (int i = 0; i < parts; ++i)
//	{
//		tt[i] = std::thread(CircleDetectionThread, DiffThreshold, CountThreshold, bnd[i], bnd[i + 1]);
//	}
//			
//	//Join parts-1 threads
//	for (int i = 0; i < parts; ++i)
//		tt[i].join();
//}

void CircleDetector::DetectCircleAt(int x, int y)
{
	int c = ImagePtr[y * W + x];
	int count = 0;
	//int count2 = 0;
	float resp = 0;
	//uint16_t resp = 0;

	for (int64_t yy = 0; yy < Mask_H; ++yy)
	{
		for (int64_t xx = 0; xx < Mask_W; ++xx)
		{
			if (MaskPtr[yy * Mask_W + xx])
			{
				//count2++;
				int64_t j = y + yy - Mask_W2;
				int64_t i = x + xx - Mask_H2;
				int v = ImagePtr[j * W + i];

				if ((c - v) >= Params.DiffThreshold)
				{
					count += 1;
					resp += (c - v);
				}
			}
		}
	}

	if (count >= Params.CountThreshold)
	{
		CircleResponsePtr[y * W + x] = resp;
	}
}

void CircleDetector::GetCenters(std::vector<Feature>& features)
{
	GetCenters(CircleResponsePtr, features, Params);
}

void CircleDetector::GetCentersFromMean(std::vector<Feature>& features)
{
	if (!Params.CalcMean)
		return;

	GetCenters(CircleResponseMeanPtr, features, Params);
}

void CircleDetector::GetCentersFromDiff(std::vector<Feature>& features)
{
	if (!Params.CalcMean)
		return;

	GetCenters(CircleResponseDiffPtr, features, Params);
}

void CircleDetector::GetCenters(uint16_t* resp_data, std::vector<Feature>& features, const DetectionParams& params)
{
	float threshold = 0;
	cv::Mat visited = cv::Mat(H, W, CV_8UC1);
	visited.setTo(0);
	uint8_t* visited_data = (uint8_t*)(visited.data);
	// funny bounds due to sampling ring radius (10) and border of previously applied blur (2)
	uint8_t label = 1;
	for (size_t y = 0; y < H; ++y)
	{
		// we output 4 pixels at a time
		for (size_t x = 0; x < W; x++)//+= 4)
		{
			size_t offset = x + y * W;

			//visited.data[offset] = 1;

			// start search
			std::vector<cv::Point2d> pts_all;
			if (resp_data[offset] > threshold)
			{
				double xsum = 0, ysum = 0, m = 0, a = 0, mean_int = 0;
				int x_min = W, x_max = 0, y_min = H, y_max = 0;

				std::list<cv::Point2d> to_visit;
				to_visit.push_back(cv::Point2d(x, y));

				while (to_visit.size() > 0)
				{
					cv::Point2d point = to_visit.front();
					to_visit.pop_front();


					if (point.x > x_max)
						x_max = static_cast<int>(point.x);
					if (point.y > y_max)
						y_max = static_cast<int>(point.y);
					if (point.x < x_min)
						x_min = static_cast<int>(point.x);
					if (point.y < y_min)
						y_min = static_cast<int>(point.y);

					size_t index = point.x + point.y * W;

					if (point.x == 17 && point.y == 318)
					{
						int bp = 1;
					}


					if (visited_data[index] != 0)
						continue;

					pts_all.push_back(point);
					visited_data[index] = label;

					xsum += point.x * resp_data[index];
					ysum += point.y * resp_data[index];
					m += resp_data[index];
					mean_int += ImagePtr[index];
					a++;

					if (point.x + 1 < W)
					{
						size_t offset1 = (point.x + 1) + point.y * W;

						if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
							to_visit.push_back(cv::Point2d(point.x + 1, point.y));
					}

					if (point.x - 1 > -1)
					{
						size_t offset1 = (point.x - 1) + point.y * W;

						if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
							to_visit.push_back(cv::Point2d(point.x - 1, point.y));
					}

					if (point.y + 1 < H)
					{
						size_t offset1 = (point.x) + (point.y + 1) * W;

						if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
							to_visit.push_back(cv::Point2d(point.x, point.y + 1));
					}

					if (point.y - 1 > -1)
					{
						size_t offset1 = (point.x) + (point.y - 1) * W;

						if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
							to_visit.push_back(cv::Point2d(point.x, point.y - 1));
					}

					if (params.EightConnected)
					{
						if (point.x + 1 < W && point.y + 1 < H)
						{
							size_t offset1 = (point.x + 1) + (point.y + 1) * W;

							if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
								to_visit.push_back(cv::Point2d(point.x + 1, point.y + 1));
						}

						if (point.x + 1 < W && point.y - 1 > -1)
						{
							size_t offset1 = (point.x + 1) + (point.y - 1) * W;

							if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
								to_visit.push_back(cv::Point2d(point.x + 1, point.y - 1));
						}

						if (point.x - 1 > -1 && point.y + 1 < H)
						{
							size_t offset1 = (point.x - 1) + (point.y + 1) * W;

							if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
								to_visit.push_back(cv::Point2d(point.x - 1, point.y + 1));
						}

						if (point.x - 1 > -1 && point.y - 1 > -1)
						{
							size_t offset1 = (point.x - 1) + (point.y - 1) * W;

							if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
								to_visit.push_back(cv::Point2d(point.x - 1, point.y - 1));
						}
					}
				}

				double density = m / a;
				if (m > params.MassMin && a > params.AreaMin && density > params.DensityMin)
					//&& x_max - x_min < params.WidthMax && y_max - y_min < params.HeightMax)
				{
					//int w = x_max - x_min;
					//int h = y_max - y_min;
					if (x_max - x_min < params.WidthMax && y_max - y_min < params.HeightMax)
					{
						//float zone_width = x_max - x_min + 1;
						//float zone_height = y_max - y_min + 1;					

						double x_out = xsum / m;
						double y_out = ysum / m;
						mean_int /= a;
						double orient = 0;

						//if (y_out > 120)
						if (x_out > params.XMin && x_out < params.XMax && y_out > params.YMin && y_out < params.YMax)
						{
							Feature tp(x_out, y_out, 0.0, a, m, orient, x_min, y_min, x_max, y_max, mean_int);
							features.push_back(tp);

							/*std::cout << "points: " << pts_all.size() << std::endl;
							for (cv::Point2d pt : pts_all)
							{
								std::cout << pt << " ";
							}
							std::cout << std::endl;*/
						}
					}
				}
			}
		}
	}
}
