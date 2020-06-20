#ifndef MNIST_H_
#define MNIST_H_

#include "../model/classification.h"
//#include "../model/network.h"
#include "../utils/read_data.h"
#include "../utils/image.h"
#include "../utils/global.h"

#include "opencv2/opencv.hpp"

namespace mnist {

int train();
void camera();

}

#endif /* MNIST_H_ */
