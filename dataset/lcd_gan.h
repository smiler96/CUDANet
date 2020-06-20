/*
* lcd_gan.h
*
*  Created on: 1 11, 2020
*      Author: ljy
*/

#ifndef LCD_GAN_H_
#define LCD_GAN_H_

#include "../model/gan.h"
#include "../model/reconstruction.h"
#include "../model/network.h"
#include "../utils/utils.h"
#include "../utils/image.h"
#include "../utils/global.h"

#include "opencv2/opencv.hpp"

namespace lcd_gan {

	int train();

}

#endif // LCD_GAN_H_