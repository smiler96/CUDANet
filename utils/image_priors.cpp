#include "image_priors.h"

namespace net_utils
{
	Prior::Prior()
	{

	}

	Prior::~Prior()
	{

	}

	void Prior::extract(const uint8_t* _in_data, float* _out_data, const int _prior_num, const int _height, const int _width, const int _channel, const int _in_num, 
		std::string _prior_save_path)
	{
		int64_t in_data_dim = _height * _width * _channel;
		int64_t out_data_dim = in_data_dim * (_prior_num + 1);

		int64_t in_offset = 0;
		int64_t out_offset = 0;

		// 初始化图像数据
		cv::Mat Img, LBPMap, CannyMap, LoGMap;
		if (_channel == 1)
		{
			Img = cv::Mat(_width, _height, CV_8UC1);
			LBPMap = cv::Mat(_width, _height, CV_8UC1);
			CannyMap = cv::Mat(_width, _height, CV_8UC1);
			LoGMap = cv::Mat(_width, _height, CV_16SC1);
		}
		else
		{
			Img = cv::Mat(_width, _height, CV_8UC3);
			LBPMap = cv::Mat(_width, _height, CV_8UC3);
			CannyMap = cv::Mat(_width, _height, CV_8UC3);
			LoGMap = cv::Mat(_width, _height, CV_16SC3);
		}

		for (int i = 0; i < _in_num; ++i)
		{
			Img = cv::Mat(_width, _height, CV_8UC1);
			LBPMap = cv::Mat(_width, _height, CV_8UC1);
			CannyMap = cv::Mat(_width, _height, CV_8UC1);
			LoGMap = cv::Mat(_width, _height, CV_16SC1);

			std::memcpy(Img.data, _in_data + in_offset, sizeof(uint8_t) * in_data_dim);

			// 先对图像进行均值滤波平滑处理
			cv::blur(Img, Img, cv::Size(3, 3));

			// 提取 Canny edge 先验信息
			cv::Canny(Img, CannyMap, 80, 160, 3);

			// 提取 Local binary Pattern 先验信息
			lbp::OLBP(Img, LBPMap);
			cv::normalize(LBPMap, LBPMap, 0, 255, NORM_MINMAX);

			// 提取 Laplacian of Gaussian 先验信息
			if(_channel == 1) cv::Laplacian(Img, LoGMap, CV_16SC1);
			else cv::Laplacian(Img, LoGMap, CV_16SC3);
			cv::normalize(LoGMap, LoGMap, 0, 255, NORM_MINMAX);
			/*LoGMap.convertTo(LoGMap, CV_8UC1);*/


			Img.convertTo(Img, CV_32FC1);
			//Img /= 255.0;
			CannyMap.convertTo(CannyMap, CV_32FC1);
			//CannyMap /= 255.0;
			LBPMap.convertTo(LBPMap, CV_32FC1);
			//LBPMap /= 255.0;
			LoGMap.convertTo(LoGMap, CV_32FC1);
			//LoGMap /= 255.0;

			if (_prior_save_path.size() > 0)
			{
				cv::imwrite(_prior_save_path + std::to_string(i) + "_img.bmp", Img);
				cv::imwrite(_prior_save_path + std::to_string(i) + "_canny.bmp", CannyMap);
				cv::imwrite(_prior_save_path + std::to_string(i) + "_lbp.bmp", LBPMap);
				cv::imwrite(_prior_save_path + std::to_string(i) + "_log.bmp", LoGMap);
			}

			if (_prior_num == 3)
			{
				std::memcpy(_out_data + out_offset, Img.data, sizeof(float) * in_data_dim);
 				std::memcpy(_out_data + out_offset + in_data_dim, CannyMap.data, sizeof(float) * in_data_dim);
				std::memcpy(_out_data + out_offset + 2 * in_data_dim, LBPMap.data, sizeof(float) * in_data_dim);
				std::memcpy(_out_data + out_offset + 3 * in_data_dim, LBPMap.data, sizeof(float) * in_data_dim);
			}

			in_offset += in_data_dim;
			out_offset += out_data_dim;
		}
	}
}