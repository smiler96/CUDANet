/*
 * layer.h
 *
 * Layer class, this class is abstract, it provides basic layer members like
 * data and some methods. All layers should extend this class.
 * 20191210, by wanqian
 *
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "../utils/global.h"
#include "../utils/set_value.h"
#include "../utils/print.h"
#include "../utils/utils.h"

namespace layer {

class Layer {
public:
	Layer(float alpha = 0, float momentum = 0.9f, float weight_decay = 0);
	virtual ~Layer();

	// three virtual method that all layers should have
	virtual void forward(bool train = true) = 0;
	virtual void backward() = 0;
	virtual void update() = 0;

	void adjust_learning(float scale); // change the learning rate

	Layer* prev; // previous layer
	Layer* next; // next layer
	
	std::string layerName;
	int layerId;

	cudnnTensorDescriptor_t t_data; // output dimension
	float* data; // output
	int data_size; // output size
	float* diff; // differential for the previous layer
	float* param; // parameters
	int param_size; // parameters count
	float* param_bias; // bias parameters for some layers
	int param_bias_size; // bias parameters count
	float* gradient; // gradient of parameters
	float* gradient_bias; // gradient of bias parameters
	int batch; // batch size
	float alpha; // learning rate
	float momentum; // momentum of gradient
	float weight_decay; // weight decay rate

	// adam —µ¡∑À„∑®
	float* para_moment1; // first moment vector
	float* para_moment2; // second moment vector
	float* gradient2;
	float* gradient3;

	float* bias_moment1; // first moment vector
	float* bias_moment2; // second moment vector
	float* bias_gradient2;
	float* bias_gradient3;

	
	bool branch = false;

	void setLayerName(std::string _name);
	void setLayerId(int _id);
	std::string getLayerName();
	int getLayerId();

	bool isBranch();
	void setBranch();
};

} /* namespace layer */
#endif /* LAYER_H_ */
