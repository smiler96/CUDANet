#ifndef _H_DECONVOLUTION_H_
#define _H_DECONVOLUTION_H_

#include "layer.h"

namespace layer {

	class Deconvolution : public Layer {
	public:
		cudnnFilterDescriptor_t filter_desc; // response for gradient in Layer
		cudnnConvolutionDescriptor_t conv_desc;
		cudnnTensorDescriptor_t bias_desc; // response for gradient_bias in Layer
		cudnnConvolutionFwdAlgo_t fwd_algo; // forward algorithm
		cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo; // backrward filter algorithm
		cudnnConvolutionBwdDataAlgo_t bwd_data_algo; // backrward data algorithm
		size_t workspace_fwd_size; // extra size for forward computing
		size_t workspace_bwd_filter_size; // extra size for backrward filter computing
		size_t workspace_bwd_data_size; // extra size for backrward data computing
		void* workspace_fwd;  
		void* workspace_bwd_filter; 
		void* workspace_bwd_data;  
		
	public:
		Deconvolution(Layer* _prev, int n, int c, int kernel, int stride, int padding, float alpha,
			float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);
		virtual ~Deconvolution();
		void forward(bool train = true);
		void backward();
		void update();
	};

} /* namespace layer */
#endif /* _H_DECONVOLUTION_H_ */
