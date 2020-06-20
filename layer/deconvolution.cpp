#include "deconvolution.h"

namespace layer {

	Deconvolution::Deconvolution(Layer* _prev, int n, int c, int kernel, int stride, int padding, float alpha,
		float sigma, float momentum, float weight_decay) :
		Layer(alpha, momentum, weight_decay) {
		prev = _prev;
		prev->next = this;

		batch = n;

		// set convolution operation descriptor
		callCudnn(cudnnCreateConvolutionDescriptor(&conv_desc));
		// s=2, p=1, k=4, double size of input data
		callCudnn(cudnnSetConvolution2dDescriptor(
			conv_desc,
			padding, padding,  // padding
			stride, stride,  //stride
			1, 1,
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT));

		int _n, _c, _h, _w, _tmp;
		cudnnDataType_t _t;
		callCudnn(cudnnGetTensor4dDescriptor(
			prev->t_data, &_t, 
			&_n, &_c, &_h, &_w, 
			&_tmp, &_tmp, &_tmp, &_tmp));

		// set convolution filter descriptor
		callCudnn(cudnnCreateFilterDescriptor(&filter_desc));
		callCudnn(cudnnSetFilter4dDescriptor(
			filter_desc,
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NCHW,
			_c, c, // reversed aganist convolution, 与卷积操作时是相反的过程
			kernel,
			kernel));

		param_size = c * _c * kernel * kernel;
		callCuda(cudaMalloc(&param, sizeof(float) * param_size)); // 卷积核参数需要的空间
		callCuda(cudaMalloc(&gradient, sizeof(float) * param_size)); // 卷积核参数梯度需要的空间

		//utils::printGpuMatrix(param, param_size, _c * kernel, c * kernel, 8);

		if (kernel < 2 * padding)
		{
			std::cout << "kernel size doesnot match padding\n";
			//system("pause");
		}

		int h = stride * (_h - 1) + kernel - 2 * padding;
		int w = stride * (_w - 1) + kernel - 2 * padding;

		if ((kernel - 2 * padding) % (stride) != 0)
		{
			h += 1;
			w += 1;
		}
		// set convolution Iutput (or deconvolution Output) data descriptor
		callCudnn(cudnnCreateTensorDescriptor(&t_data));
		callCudnn(cudnnSetTensor4dDescriptor(
			t_data,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c, h, w));

		data_size = n * c * h * w;
		callCuda(cudaMalloc(&data, sizeof(float) * data_size));   // 输出数据需要的空间
		callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size));  // 梯度回传需要的空间

		net_utils::setGpuValue(data, data_size, 0);
		net_utils::setGpuValue(diff, prev->data_size, 0);

		// set bias data descriptor
		callCudnn(cudnnCreateTensorDescriptor(&bias_desc));
		callCudnn(cudnnSetTensor4dDescriptor(
			bias_desc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			1, c, 1, 1));  

		param_bias_size = c;
		callCuda(cudaMalloc(&param_bias, sizeof(float) * param_bias_size)); // 卷积核偏差参数需要的空间
		callCuda(cudaMalloc(&gradient_bias, sizeof(float) * param_bias_size));  // 卷积核偏差参数梯度需要的空间

		// set parameters and gradients initial values
		net_utils::setGpuNormalValue(param, param_size, 0, sigma);
		net_utils::setGpuNormalValue(param_bias, param_bias_size, 0, sigma);
		net_utils::setGpuValue(gradient, param_size, 0);
		net_utils::setGpuValue(gradient_bias, param_bias_size, 0);

		// initialize all to default algorithms
		fwd_algo = (cudnnConvolutionFwdAlgo_t)0;
		bwd_filter_algo = (cudnnConvolutionBwdFilterAlgo_t)0;
		bwd_data_algo = (cudnnConvolutionBwdDataAlgo_t)0;	

		// get convolution forward operation algorithm 
		// t_data * filter_desc = prev->t_data
		callCudnn(cudnnGetConvolutionForwardAlgorithm(
			global::cudnnHandle,
			t_data,   // this layer's output data descriptor
			filter_desc,  // this layer's convolution filter descriptor
			conv_desc,  // this layer's convolution operation descriptor
			prev->t_data,	 // previous layer's convolution output data descriptor
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, //CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 
			global::workspace_limit_bytes, //0, 
			&fwd_algo));
		// get workspace for forkward deconvolution algorithm
		callCudnn(cudnnGetConvolutionForwardWorkspaceSize(
			global::cudnnHandle,
			t_data,
			filter_desc,
			conv_desc,
			prev->t_data, // previous layer's descriptor
			fwd_algo,
			&workspace_fwd_size));
		callCuda(cudaMalloc(&workspace_fwd, workspace_fwd_size));

		// choose backward algorithm for filter
		callCudnn(cudnnGetConvolutionBackwardFilterAlgorithm(
			global::cudnnHandle,
			t_data,
			prev->t_data,
			conv_desc,
			filter_desc,
			CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
			global::workspace_limit_bytes,
			&bwd_filter_algo));
		// get workspace for backward deconvolution filter algorithm
		callCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
			global::cudnnHandle,
			t_data,
			prev->t_data,  // previous layer's descriptor
			conv_desc,
			filter_desc,
			bwd_filter_algo,
			&workspace_bwd_filter_size));
		callCuda(cudaMalloc(&workspace_bwd_filter, workspace_bwd_filter_size)); // 已经经过修改

		// choose backward algo for data
		callCuda(cudnnGetConvolutionBackwardDataAlgorithm(
			global::cudnnHandle,
			filter_desc,
			prev->t_data,
			conv_desc,
			t_data,
			CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
			global::workspace_limit_bytes,
			&bwd_data_algo));
		// get workspace for back data
		callCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
			global::cudnnHandle,
			filter_desc,
			prev->t_data,  // previous layer's descriptor
			conv_desc,
			t_data,
			bwd_data_algo,
			&workspace_bwd_data_size));
		callCuda(cudaMalloc(&workspace_bwd_data, workspace_bwd_data_size));

	}

	Deconvolution::~Deconvolution() {
		callCudnn(cudnnDestroyFilterDescriptor(filter_desc));
		callCudnn(cudnnDestroyConvolutionDescriptor(conv_desc));
		callCudnn(cudnnDestroyTensorDescriptor(t_data));
		callCudnn(cudnnDestroyTensorDescriptor(bias_desc));
		callCuda(cudaFree(data));
		callCuda(cudaFree(diff));
		callCuda(cudaFree(param));
		callCuda(cudaFree(param_bias));
		callCuda(cudaFree(gradient));
		callCuda(cudaFree(gradient_bias));
		callCuda(cudaFree(workspace_fwd));
		callCuda(cudaFree(workspace_bwd_filter));
		callCuda(cudaFree(workspace_bwd_data));
	}

	void Deconvolution::forward(bool train) {
		float a = 1;
		float b = 0;
		callCudnn(cudnnConvolutionBackwardData(
			global::cudnnHandle,
			&a,
			filter_desc, param,
			prev->t_data, prev->data,
			conv_desc,
			bwd_data_algo,
			workspace_bwd_data,
			workspace_bwd_data_size,
			&b,
			t_data, data));
		callCudnn(cudnnAddTensor(
			global::cudnnHandle,
			&a,
			bias_desc, param_bias,
			&a,
			t_data, data)); // this layer's output data add bias
	}

	void Deconvolution::backward() {
		//float a = alpha; // learning rate
		float a = 1; // learning rate
		//float b = 0; // momentum
		float b = momentum;
		callCudnn(cudnnConvolutionBackwardBias(
			global::cudnnHandle,
			&a,
			t_data,
			next->diff,
			&b,
			bias_desc, gradient_bias));
		callCudnn(cudnnConvolutionBackwardFilter(
			global::cudnnHandle,
			&a,
			t_data, next->diff,
			prev->t_data, prev->data,
			conv_desc,
			bwd_filter_algo,
			workspace_bwd_filter, workspace_bwd_filter_size,
			&b,
			filter_desc, gradient));
		
		a = 1;
		b = 0;
		callCudnn(cudnnConvolutionForward(
			global::cudnnHandle,
			&a,
			t_data, next->diff,
			filter_desc, param,
			conv_desc,
			fwd_algo,
			workspace_fwd, workspace_fwd_size,
			&b,
			prev->t_data, diff));
	}

	void Deconvolution::update() {
		float a = 1 - weight_decay;
		// update the convolution filter parameters
		callCuda(cublasSaxpy(
			global::cublasHandle,
			param_size,
			&a,
			gradient,
			1,
			param,
			1));

		// update the convolution filter bias parameters
		callCuda(cublasSaxpy(
			global::cublasHandle,
			param_bias_size,
			&a,
			gradient_bias,
			1,
			param_bias,
			1));
	}

}