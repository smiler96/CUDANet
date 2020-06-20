/*
 * utils.h
 *
 *  Created on: Sep 20, 2019
 *      Author: wanqian
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <io.h>
#include <string>
#include <direct.h>
#include <list>

#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2\opencv.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkValidNum(x) do {                                          \
    std::stringstream _error;                                          \
    if (isnan(x) || isinf(x)) {											\
    	_error << "Nan/Inf fatal error: " << x;							\
    	FatalError(_error.str());                                      \
    }                                                                  \
} while(0)

#define callCudnn(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
    	_error << "CuDNN failure: " << cudnnGetErrorString(status);    \
    	FatalError(_error.str());                                      \
    }                                                                  \
} while(0)

#define callCurand(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CURAND_STATUS_SUCCESS) {                             \
    	_error << "CuRAND failure: " << status;   					   \
    	FatalError(_error.str());                                      \
    }                                                                  \
} while(0)

#define callCuda(status) do {                                  		   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
    	_error << "Cuda failure: " << status;                          \
    	FatalError(_error.str());                                      \
    }                                                                  \
} while(0)



namespace net_utils {

	void getAllFiles(std::string path, std::vector<std::string>& files);
	void getAllFormatFiles(std::string path, std::vector<std::string>& files, std::string format);
	
	void getAllImages(uint8_t* images, const std::vector<std::string> files, const int data_dim);

	void checkArrayNan(const float* x, const int n);

} /* namespace utils */
#endif /* UTILS_H_ */
