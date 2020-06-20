#include "activation.h"

using namespace global;

namespace layer {

Activation::Activation(Layer* _prev, cudnnActivationMode_t _mode) : Layer(0) {
	prev = _prev;
	prev->next = this;

	mode = _mode;
	callCudnn(cudnnCreateActivationDescriptor(&activation_descriptor));
	callCudnn(cudnnSetActivationDescriptor(activation_descriptor,
		mode,
		CUDNN_PROPAGATE_NAN,
		/*relu_coef=*/0));

	int _n, _c, _h, _w, _tmp;
	cudnnDataType_t _t;
	callCudnn(cudnnGetTensor4dDescriptor(prev->t_data, &_t, &_n, &_c, &_h, &_w, &_tmp,
			&_tmp, &_tmp, &_tmp));
	batch = _n;
	data_size = _n * _c * _h * _w;
	callCudnn(cudnnCreateTensorDescriptor(&t_data));
	callCudnn(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			_n, _c, _h, _w));
	callCuda(cudaMalloc(&data, sizeof(float) * data_size));
	callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size));

	net_utils::setGpuValue(data, data_size, 0);
	net_utils::setGpuValue(diff, prev->data_size, 0);

	param_size = 0;
	param_bias_size = 0;
}

Activation::~Activation() {
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCuda(cudaFree(data));
	callCuda(cudaFree(diff));
}

void Activation::forward(bool train) {
	float a = 1;
	float b = 0;
	callCudnn(cudnnActivationForward(
		cudnnHandle, 
		activation_descriptor, 
		&a, 
		prev->t_data, 
		prev->data, 
		&b, 
		t_data,
		data));
}

void Activation::backward() {
	float a = 1;
	float b = 0;
	callCudnn(cudnnActivationBackward(
		cudnnHandle, 
		activation_descriptor,
		&a, 
		t_data, data,
		t_data, next->diff, 
		prev->t_data, prev->data, 
		&b, 
		t_data, diff));
}

void Activation::update() {
	// nothing
}

} /* namespace layer */
