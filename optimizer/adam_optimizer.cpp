#include "adam_optimizer.h"

namespace optimizer
{
	Adam::Adam(float _learn_rate, float _weight_decay, float _beta1, float _beta2, float _epsilon) :
		Optimizer(_learn_rate, _weight_decay)
	{
		beta1 = _beta1;
		beta2 = _beta2;
		invbeta1 = 1 - beta1;
		invbeta2 = 1 - beta2;
		epsilon = _epsilon;

		inv_weight_decay = 1 - weight_decay;
	}

	Adam::~Adam()
	{

	}

	void Adam::adjust_lr(float _step_decay)
	{
		/*learn_rate *= _step_decay;
		if (learn_rate < 1e-5) learn_rate = 1e-5;*/
	}

	void Adam::optimize(Layer* layer, const int t)
	{
		
		int64 _len = 0;
		m1_hat = 1.0 / (1 - std::pow(beta1, t));
		m2_hat = 1.0 / (1 - std::pow(beta2, t));

		checkValidNum(m1_hat);
		checkValidNum(m2_hat);

		// update the convolution filter parameters
		if (layer->param_size > 0)
		{
			_len = layer->param_size;
			if (!(layer->para_moment1))
			{
				// adam 训练一阶、二阶动量赋值 初始化
				callCuda(cudaMalloc(&(layer->para_moment1), sizeof(float) * _len));
				callCuda(cudaMemset(layer->para_moment1, 0, sizeof(float) * _len));
				callCuda(cudaMalloc(&(layer->para_moment2), sizeof(float) * _len));
				callCuda(cudaMemset(layer->para_moment2, 0, sizeof(float) * _len));
				// gradient^2
				callCuda(cudaMalloc(&(layer->gradient2), sizeof(float) * _len));
				// m1/(1-β1^t) / (sqrt(m2/(1-β2^t) + epsilon))
				callCuda(cudaMalloc(&(layer->gradient3), sizeof(float) * _len));
			}

			callCuda(cublasSscal(global::cublasHandle, _len, &beta1, layer->para_moment1, 1));
			callCuda(cublasSaxpy(global::cublasHandle, _len, &invbeta1, layer->gradient, 1, layer->para_moment1, 1));
			
			// gradient^2
			callCuda(cudaMemset(layer->gradient2, 0, sizeof(float) *_len));
			func_gpu::powGpu(layer->gradient, _len, 2, 1, 0, layer->gradient2);

			callCuda(cublasSscal(global::cublasHandle, _len, &beta2, layer->para_moment2, 1));
			callCuda(cublasSaxpy(global::cublasHandle, _len, &invbeta2, layer->gradient2, 1, layer->para_moment2, 1));
			
			func_gpu::adamUpdateGpu(layer->para_moment1, layer->para_moment2, _len, t, beta1, beta2, epsilon, layer->gradient3);

			// update: para = (1-weight_decay)*para + learn_rate*gradient3   learn_rate 是负数
			callCuda(cublasSscal(global::cublasHandle, _len, &inv_weight_decay, layer->param, 1));
			callCuda(cublasSaxpy(global::cublasHandle, _len, &learn_rate, layer->gradient3, 1, layer->param, 1));
		}
		
		// 更新 偏差参数
		if (layer->param_bias_size > 0)
		{
			_len = layer->param_bias_size;

			if (!(layer->bias_moment1))
			{
				// adam 训练一阶、二阶动量赋值
				callCuda(cudaMalloc(&(layer->bias_moment1), sizeof(float) * _len));
				callCuda(cudaMemset(layer->bias_moment1, 0, sizeof(float) * _len));
				callCuda(cudaMalloc(&(layer->bias_moment2), sizeof(float) * _len));
				callCuda(cudaMemset(layer->bias_moment2, 0, sizeof(float) * _len));

				callCuda(cudaMalloc(&(layer->bias_gradient2), sizeof(float) * _len));
				callCuda(cudaMalloc(&(layer->bias_gradient3), sizeof(float) * _len));
			}

			// 更新一阶动量 m1 = β1*m1 + (1-β1)*gt
			callCuda(cublasSscal(global::cublasHandle, _len, &beta1, layer->bias_moment1, 1));
			callCuda(cublasSaxpy(global::cublasHandle, _len, &invbeta1, layer->gradient_bias, 1, layer->bias_moment1, 1));

			// gradient^2
			callCuda(cudaMemset(layer->bias_gradient2, 0, sizeof(float) * _len));
			func_gpu::powGpu(layer->gradient_bias, _len, 2, 1, 0, layer->bias_gradient2);
	
			// 更新二阶动量 m2 = β2*m2 + (1-β2)*gt^2
			callCuda(cublasSscal(global::cublasHandle, _len, &beta2, layer->bias_moment2, 1));
			callCuda(cublasSaxpy(global::cublasHandle, _len, &invbeta2, layer->bias_gradient2, 1, layer->bias_moment2, 1));

			func_gpu::adamUpdateGpu(layer->bias_moment1, layer->bias_moment2, _len, t, beta1, beta2, epsilon, layer->bias_gradient3);

			// update: para = para + learn_rate*gradient3   learn_rate 是负数
			callCuda(cublasSscal(global::cublasHandle, _len, &inv_weight_decay, layer->param_bias, 1));
			callCuda(cublasSaxpy(global::cublasHandle, _len, &learn_rate, layer->bias_gradient3, 1, layer->param_bias, 1));
		}
	}

}
