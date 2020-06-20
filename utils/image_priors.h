/*
* image_priors.h
*
* extract image priors, such as canny LBP LOG
*
*  Created on: 1 11, 2020
*      Author: wanqian
*/

#ifndef _IMAGE_PRIORS_H_
#define _IMAGE_PRIORS_H_

#include <stdint.h>
#include "image.h"
#include "lbp.hpp"

namespace net_utils 
{
	class Prior
	{
	public:
		Prior();
		~Prior();

		/*
		* extract image priors
		*
		* _in_data: pointer of input images
		* _out_data: pointer of output images with priors
		* _prior_num: prior numbers
		* _height: input image's height
		* _width: input image's width
		* _channel: input image's channel
		* _in_num: input images' numbers
		*/
		static void extract(const uint8_t* _in_data, float* _out_data, const int _prior_num, const int _height, const int _width, const int _channel, const int _in_num, 
			std::string _prior_save_path);
	};

}

#endif // !_IMAGE_PRIORS_H_
