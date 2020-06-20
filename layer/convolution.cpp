#include "convolution.h"

using namespace global;

namespace layer {

Convolution::Convolution(Layer* _prev, int n ,int c, int kernel, int stride, int padding, float alpha,
		float sigma, float momentum, float weight_decay):
		Layer(alpha, momentum, weight_decay) 
{
	prev = _prev;
	prev->next = this;

	batch = n;

	// set convolution operation descriptor
	callCudnn(cudnnCreateConvolutionDescriptor(&descriptor));
	callCudnn(cudnnSetConvolution2dDescriptor(descriptor, 
		padding, padding,  // zero-padding
		stride, stride,  //stride
		1, 1,
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT));

	int _n, _c, _h, _w, _tmp;
	cudnnDataType_t _t;
	callCudnn(cudnnGetTensor4dDescriptor(prev->t_data, &_t, &_n, &_c, &_h, &_w, &_tmp,
			&_tmp, &_tmp, &_tmp));

	// set convolution filter descriptor
	callCudnn(cudnnCreateFilterDescriptor(&filter_desc));
	callCudnn(cudnnSetFilter4dDescriptor(filter_desc,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		c, _c, 
		kernel,
		kernel));
	param_size =  _c * c * kernel * kernel;
	callCuda(cudaMalloc(&param, sizeof(float) * param_size));
	callCuda(cudaMalloc(&gradient, sizeof(float) * param_size));

	//utils::printGpuMatrix(param, param_size, _c * kernel, c * kernel, 8);

	int h = std::ceil((_h - kernel + 2 * padding) / stride) + 1; // 取下界
	int w = std::ceil((_w - kernel + 2 * padding) / stride) + 1;

	// set convolution output data descriptor
	callCudnn(cudnnCreateTensorDescriptor(&t_data));
	callCudnn(cudnnSetTensor4dDescriptor(t_data, 
		CUDNN_TENSOR_NCHW,	
		CUDNN_DATA_FLOAT,
		n, 
		c,
		h,
		w));
	data_size = n * c * h * w;
	callCuda(cudaMalloc(&data, sizeof(float) * data_size));
	callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size)); // 梯度回传需要的数据空间

	net_utils::setGpuValue(data, data_size, 0);
	net_utils::setGpuValue(diff, prev->data_size, 0);

	// set bias data descriptor
	callCudnn(cudnnCreateTensorDescriptor(&t_bias));
	callCudnn(cudnnSetTensor4dDescriptor(t_bias, 
		CUDNN_TENSOR_NCHW,	
		CUDNN_DATA_FLOAT,
		1, 
		c, 
		1, 
		1));
	param_bias_size =  c;
	callCuda(cudaMalloc(&param_bias, sizeof(float) * param_bias_size));
	callCuda(cudaMalloc(&gradient_bias, sizeof(float) * param_bias_size));

	net_utils::setGpuNormalValue(param, param_size, 0, sigma);
	net_utils::setGpuNormalValue(param_bias, param_bias_size, 0, sigma);
	net_utils::setGpuValue(gradient, param_size, 0);
	net_utils::setGpuValue(gradient_bias, param_bias_size, 0);
	
	// get convolution forward operation algorithm 
	callCudnn(cudnnGetConvolutionForwardAlgorithm(
		cudnnHandle, 
		prev->t_data,  // previous layer's output data descriptor
		filter_desc, // this layer's convolution filter descriptor
		descriptor, // this layer's convolution operation descriptor
		t_data,	 // this layer's convolution output data descriptor
		CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, //CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 
		global::workspace_limit_bytes, //0, 
		&algo));

	/* add this from caffe cudnn_deconv_layer.cpp */
	// We have found that CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM is
	// buggy. Thus, if this algo was chosen, choose winograd instead. If
	// winograd is not supported or workspace is larger than threshold, choose
	// implicit_gemm instead.
	if (algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
		size_t winograd_workspace_size;
		cudnnStatus_t status = cudnnGetConvolutionForwardWorkspaceSize(
			cudnnHandle,
			prev->t_data,
			filter_desc,
			descriptor,
			t_data,
			CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
			&winograd_workspace_size);
		if (status != CUDNN_STATUS_SUCCESS ||
			winograd_workspace_size >= workspace_limit_bytes) {
			algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
		}
		else {
			algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
		}
	}

	// get workspace for forkward algorithm
	callCudnn(cudnnGetConvolutionForwardWorkspaceSize(
		cudnnHandle,
		prev->t_data, // previous layer's descriptor
		filter_desc,
		descriptor, 
		t_data,
		algo, 
		&workspace_size));
	callCuda(cudaMalloc(&workspace, workspace_size));

	// choose backward algorithm for filter
	callCudnn(cudnnGetConvolutionBackwardFilterAlgorithm(
		cudnnHandle,
		prev->t_data, // previous layer's descriptor
		t_data,
		descriptor,
		filter_desc,
		CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
		workspace_limit_bytes,
		&bwd_filter_algo));

	// get workspace for backwards filter algorithm
	callCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		cudnnHandle,
		prev->t_data, // previous layer's descriptor
		t_data,
		descriptor,
		filter_desc,
		bwd_filter_algo,
		&workspace_bwd_filter_size));
	callCuda(cudaMalloc(&workspacefilter, workspace_bwd_filter_size)); // 已经经过修改

	// choose backward algo for data
	callCudnn(cudnnGetConvolutionBackwardDataAlgorithm(
		cudnnHandle,
		filter_desc,
		t_data,
		descriptor,
		prev->t_data, // previous layer's descriptor
		CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
		workspace_limit_bytes,
		&bwd_data_algo));

	// get workspace size for back data
	callCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
		cudnnHandle,
		filter_desc,
		t_data,
		descriptor,
		prev->t_data,
		bwd_data_algo,
		&workspace_bwd_data_size));
	callCuda(cudaMalloc(&workspacebackdata, workspace_bwd_data_size));

}

Convolution::~Convolution() {
	callCudnn(cudnnDestroyFilterDescriptor(filter_desc));
	callCudnn(cudnnDestroyConvolutionDescriptor(descriptor));
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCudnn(cudnnDestroyTensorDescriptor(t_bias));
	callCuda(cudaFree(data));
	callCuda(cudaFree(diff));
	callCuda(cudaFree(param));
	callCuda(cudaFree(param_bias));
	callCuda(cudaFree(gradient));
	callCuda(cudaFree(gradient_bias));
	callCuda(cudaFree(workspace));
	callCuda(cudaFree(workspacefilter));
	callCuda(cudaFree(workspacebackdata));
}

void Convolution::forward(bool train) {
	float a = 1;
	float b = 0;
	//net_utils::printGpuMatrix(param, param_size, 4, 4, 5);
	callCudnn(cudnnConvolutionForward(
		cudnnHandle, 
		&a, 
		prev->t_data, prev->data,  // previous layer's output data
		filter_desc, param,   // this layer's fliter's weight parameters
		descriptor,   // this layer's operation(convolution) descriptor
		algo,   // this layer's operation(convolution) implement algorithm
		workspace, workspace_size, 
		&b, 
		t_data, data)); // this layer's output data
	callCudnn(cudnnAddTensor(
		cudnnHandle, 
		&a, 
		t_bias, param_bias,
		&a, 
		t_data, data)); // this layer's output data add bias
}

void Convolution::backward() {
	//float a = alpha; // learning rate
	float a = 1; // learning rate
	//float b = 0; // momentum
	float b = momentum; // momentum
	callCudnn(cudnnConvolutionBackwardBias(
		cudnnHandle, 
		&a, 
		t_data,
		next->diff, // next layer's differential propagate to this layer
		&b,
		t_bias, gradient_bias));
	callCudnn(cudnnConvolutionBackwardFilter(
		cudnnHandle, 
		&a, 
		prev->t_data, prev->data, 
		t_data, next->diff,
		descriptor, 
		bwd_filter_algo, //CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
		workspacefilter, workspace_bwd_filter_size,
		&b, 
		filter_desc, gradient));

	a = 1;
	b = 0;
	callCudnn(cudnnConvolutionBackwardData(
		cudnnHandle, 
		&a, 
		filter_desc, param,
		t_data, 
		next->diff, 
		descriptor, 
		bwd_data_algo, //CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
		workspacebackdata, workspace_bwd_data_size,
		&b, 
		prev->t_data, 
		diff));
}

void Convolution::update() {

	/*float beta1 = 0.99;
	float _beta1 = 1 - beta1;
	float beta2 = 0.999;
	float _beta2 = 1 - beta2;
	float eps = 1e-8;
	callCuda(cublasSaxpy(cublasHandle, param_size, &beta1, para_moment1, 1, para_moment1, 1)); 
	callCuda(cublasSaxpy(cublasHandle, param_size, &_beta1, gradient, 1, para_moment1, 1));*/

	float a = 1 - weight_decay;
	// update the convolution filter parameters
	callCuda(cublasSaxpy(
		cublasHandle, 
		param_size,
		&a, 
		gradient, 
		1,
		param, 
		1));

	// update the convolution filter bias parameters
	callCuda(cublasSaxpy(
		cublasHandle, 
		param_bias_size,
		&a,
		gradient_bias,
		1,
		param_bias, 
		1));
}

}
