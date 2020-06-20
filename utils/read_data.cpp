/*
 * read_data.cpp
 *
 *  Created on: 11/20, 2019
 *      Author: wanqian
 */

#include "read_data.h"

namespace net_utils {

void readInt(std::ifstream& stream, int* val) {
	// little endian
	for (int i = sizeof(int) - 1; i >= 0; i--)
		stream.read(((char*)val) + i, 1);
}

void readBytes(std::ifstream& stream, uint8_t* val, int n) {
	stream.read((char*)val, n);
}

void readFloat(std::ifstream& stream, float* val, int n) {
	stream.read((char*)val, sizeof(float) * n);
}

void readCPUMatrix(std::string dir, float* val, int n) {
	std::ifstream input(dir, std::ios::in | std::ios::binary);
	readFloat(input, val, n);
	input.close();
}

void readGPUMatrix(std::string dir, float* d_val, int n) {
	float* val = new float[n];
	readCPUMatrix(dir, val, n);
	cudaMemcpy(d_val, val, sizeof(float) * n, cudaMemcpyHostToDevice);
	delete[] val;
}

}


