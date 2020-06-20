#ifndef CONVOLUTION_CUH_
#define CONVOLUTION_CUH_

#include "layer.h"
//#include ".././utils/set_value.h"

namespace layer {

class Convolution : public Layer {
public:
	cudnnFilterDescriptor_t filter_desc;  // response for gradient in Layer
	cudnnConvolutionDescriptor_t descriptor;
	cudnnTensorDescriptor_t t_bias;  // response for gradient_bias in Layer
	cudnnConvolutionFwdAlgo_t algo;
	cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
	cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
	size_t workspace_size; // extra size for computing
	size_t workspace_bwd_filter_size; // extra size for computing
	size_t workspace_bwd_data_size; // extra size for computing
	void* workspace; // pointer to the extra size
	void* workspacefilter; // pointer to the extra size
	void* workspacebackdata; // pointer to the extra size
	float* tmp_data;
	float* tmp_diff;
public:
	Convolution(Layer* _prev, int n ,int c, int kernel, int stride, int padding, float alpha,
			float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);
	virtual ~Convolution();
	void forward(bool train = true);
	void backward();
	void update();
};

} /* namespace layer */
#endif /* CONVOLUTION_CUH_ */
