/*
 * imagenet.h
 *
 *  Created on: Oct 11, 2015
 *      Author: lyx
 */

#ifndef IMAGENET_H_
#define IMAGENET_H_

#include <vector>
#include <map>
#include <string>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <cassert>

#include "../model/classification.h"
#include "../utils/read_data.h"

#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

namespace imagenet {

int train();

}

#endif /* IMAGENET_H_ */
