/*
class BatchNorm inherits from Layer
for conv spatial batch normalization
20191219
created by wanqian
*/

#include "batch_normalization.h"

namespace layer
{

	BatchNorm::BatchNorm(Layer* _prev, int _channels, float _epsilon, float _expAverFactor, float alpha, float sigma, float momentum, float weight_decay):
		Layer(alpha, momentum, weight_decay)
	{
		prev = _prev;
		prev->next = this;

		epsilon = _epsilon;
		if (epsilon < CUDNN_BN_MIN_EPSILON) epsilon = 1e-5;
		expAverFactor = _expAverFactor;
		if (expAverFactor < 0 || expAverFactor>1) expAverFactor = 0.1;

		channels = _channels;

		int _n, _c, _h, _w, _tmp;
		cudnnDataType_t _t;
		callCudnn(cudnnGetTensor4dDescriptor(prev->t_data, &_t, &_n, &_c, &_h, &_w, &_tmp, &_tmp, &_tmp, &_tmp));

		if (channels != _c) FatalError("The feature channels must be same as the prev layer!");
		batch = _n;
		data_size = _n * _c * _h * _w;

		param_size = _c;
		param_bias_size = _c;

		bnMode = CUDNN_BATCHNORM_SPATIAL;
		callCudnn(cudnnCreateTensorDescriptor(&t_data));
		callCudnn(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _n, _c, _h, _w));

		callCudnn(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDiscriptor));
		callCudnn(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDiscriptor, t_data, bnMode));

		//cudnnTensorDescriptor_t resultMovingMeanDiscriptor;
		callCudnn(cudnnCreateTensorDescriptor(&resultMovingMeanDiscriptor));
		callCudnn(cudnnSetTensor4dDescriptor(resultMovingMeanDiscriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _c, _h, _w));
		//cudnnTensorDescriptor_t resultMovingVarDiscriptor;
		callCudnn(cudnnCreateTensorDescriptor(&resultMovingVarDiscriptor));
		callCudnn(cudnnSetTensor4dDescriptor(resultMovingVarDiscriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _c, _h, _w));
		//cudnnTensorDescriptor_t resultSaveMeanDiscriptor;
		callCudnn(cudnnCreateTensorDescriptor(&resultSaveMeanDiscriptor));
		callCudnn(cudnnSetTensor4dDescriptor(resultSaveMeanDiscriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _c, _h, _w));
		//cudnnTensorDescriptor_t resultSaveVarDiscriptor;
		callCudnn(cudnnCreateTensorDescriptor(&resultSaveVarDiscriptor));
		callCudnn(cudnnSetTensor4dDescriptor(resultSaveVarDiscriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, _c, _h, _w));

		callCuda(cudaMalloc(&data, sizeof(float) * data_size));
		callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size)); // 梯度回传需要的数据空间

		/*callCuda(cudaMalloc(&bnBias, sizeof(float) * param_size));
		callCuda(cudaMalloc(&bnScale, sizeof(float) * param_size));
		callCuda(cudaMalloc(&bnBiasDiff, sizeof(float) * param_size));
		callCuda(cudaMalloc(&bnScaleDiff, sizeof(float) * param_size));*/

		// 网络参数
		callCuda(cudaMalloc(&param_bias, sizeof(float) * param_size));
		callCuda(cudaMalloc(&param, sizeof(float) * param_size));
		callCuda(cudaMalloc(&gradient_bias, sizeof(float) * param_size));
		callCuda(cudaMalloc(&gradient, sizeof(float) * param_size));

		callCuda(cudaMalloc(&resultMovingMean, sizeof(float) * _c * _h * _w));
		callCuda(cudaMalloc(&resultMovingVar, sizeof(float) * _c * _h * _w));
		callCuda(cudaMalloc(&resultSaveMean, sizeof(float) * _c * _h * _w));
		callCuda(cudaMalloc(&resultSaveInvVar, sizeof(float) * _c * _h * _w));

		net_utils::setGpuValue(data, data_size, 0);
		net_utils::setGpuValue(diff, prev->data_size, 0);

		net_utils::setGpuValue(resultMovingMean, _c * _h * _w, 0);
		net_utils::setGpuValue(resultMovingVar, _c * _h * _w, 0);
		net_utils::setGpuValue(resultSaveMean, _c * _h * _w, 0);
		net_utils::setGpuValue(resultSaveInvVar, _c * _h * _w, 0);

		/*net_utils::setGpuNormalValue(bnBias, channels, 0, sigma);
		net_utils::setGpuNormalValue(bnScale, channels, 0, sigma);
		net_utils::setGpuValue(bnBiasDiff, channels, 0);
		net_utils::setGpuValue(bnScaleDiff, channels, 0);*/

		net_utils::setGpuNormalValue(param_bias, channels, 0, sigma);
		net_utils::setGpuNormalValue(param, channels, 0, sigma);

		net_utils::setGpuValue(gradient_bias, channels, 0);
		net_utils::setGpuValue(gradient, channels, 0);

		bnBias = param_bias;
		bnScale = param;
		bnBiasDiff = gradient_bias;
		bnScaleDiff = gradient;

	}

	void BatchNorm::forward(bool train)
	{
		float a = 1.0, b = 0;
		if (train)
		{
			callCudnn(cudnnBatchNormalizationForwardTraining(
				global::cudnnHandle, bnMode,
				&a, &b,
				prev->t_data, prev->data,
				t_data, data,
				bnScaleBiasMeanVarDiscriptor, bnScale, bnBias,
				expAverFactor, 
				resultMovingMean, resultMovingVar,
				epsilon, 
				resultSaveMean, resultSaveInvVar));
		}
		else // test 阶段
		{
			callCudnn(cudnnBatchNormalizationForwardInference(
				global::cudnnHandle, bnMode,
				&a, &b,
				prev->t_data, prev->data,
				t_data, data,
				bnScaleBiasMeanVarDiscriptor, bnScale, bnBias,
				resultMovingMean, resultMovingVar,
				epsilon));
		}
	}

	void BatchNorm::backward()
	{
		float a = 1.0, b = 0;

		//float alphaParamDiff = alpha; // learning rate
		float alphaParamDiff = 1; // learning rate
		//float betaParamDiffb = 0; // momentum
		float betaParamDiffb = momentum; // momentum

		callCudnn(cudnnBatchNormalizationBackward(
			global::cudnnHandle, bnMode,
			&a, &b, 
			&alphaParamDiff, &betaParamDiffb,
			prev->t_data, prev->data, 
			t_data, next->diff, 
			prev->t_data, diff,
			bnScaleBiasMeanVarDiscriptor, bnScale, 
			bnScaleDiff, bnBiasDiff, 
			epsilon,
			resultSaveMean, resultSaveInvVar));
	}

	void BatchNorm::update()
	{
		float a = 1 - weight_decay;
		// update the bnScale parameters
		callCuda(cublasSaxpy(cublasHandle, channels, &a, bnScaleDiff, 1, bnScale, 1));

		// update the bnBias parameters
		callCuda(cublasSaxpy(cublasHandle, channels, &a, bnBiasDiff, 1, bnBias, 1));
	}

	BatchNorm::~BatchNorm()
	{
		callCudnn(cudnnDestroyTensorDescriptor(t_data));
		
		callCudnn(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDiscriptor));

		callCudnn(cudnnDestroyTensorDescriptor(resultMovingMeanDiscriptor));
		callCudnn(cudnnDestroyTensorDescriptor(resultMovingVarDiscriptor));
		callCudnn(cudnnDestroyTensorDescriptor(resultSaveMeanDiscriptor));
		callCudnn(cudnnDestroyTensorDescriptor(resultSaveVarDiscriptor));

		callCuda(cudaFree(data));
		callCuda(cudaFree(diff));
		callCuda(cudaFree(bnBias));
		callCuda(cudaFree(bnScale));
		callCuda(cudaFree(bnBiasDiff));
		callCuda(cudaFree(bnScaleDiff));
		callCuda(cudaFree(resultMovingMean));
		callCuda(cudaFree(resultMovingVar));
		callCuda(cudaFree(resultSaveMean));
		callCuda(cudaFree(resultSaveInvVar));
	}

}
