/*
* write_data.cpp
*
*  Created on: 11/20, 2019
*      Author: wanqian
*/

#include "write_data.h"

namespace net_utils {

void writeGPUMatrix(std::string dir, float* d_val, int n) {
	float* val = new float[n];
	cudaMemcpy(val, d_val, sizeof(float) * n, cudaMemcpyDeviceToHost);
	std::ofstream out(dir, std::ios::out | std::ios::binary);
	out.write((char*)val, sizeof(float) * n);
	out.close();
	delete[] val;
}

void writeCPUMatrix(std::string dir, float* val, int n) {
	std::ofstream out(dir, std::ios::out | std::ios::binary);
	out.write((char*)val, sizeof(float) * n);
	out.close();
}

}
