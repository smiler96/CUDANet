#include "euclidean_loss.h"

namespace layer {


	EuclideanLoss::EuclideanLoss(Layer* _prev, float* _label, float _loss_weight) {
		prev = _prev;
		prev->next = this;
		batch = prev->batch;

		next = nullptr;

		label = _label; // GPU data
		//temp_data_size = prev->data_size;

		loss_weight = _loss_weight;

		/*callCudnn(cudnnCreateTensorDescriptor(&t_data));
		callCudnn(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			batch, 1, 1, 1));*/
		data_size = 1;
		// dot loss
		loss = 0.0f;
		//callCuda(cudaMalloc(&data, sizeof(float) * 1));
		callCuda(cudaMalloc(&temp_data, sizeof(float) * prev->data_size));
		callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size));

		data = new float(0);
		//net_utils::setGpuValue(data, batch, 0);
		net_utils::setGpuValue(temp_data, prev->data_size, 0);
		net_utils::setGpuValue(diff, prev->data_size, 0);

		param_size = 0;
		param_bias_size = 0;
	}

	EuclideanLoss::~EuclideanLoss() {
		//callCudnn(cudnnDestroyTensorDescriptor(t_data));
		//label = NULL;
		//callCuda(cudaFree(label));
		//callCuda(cudaFree(data));
		delete data;
		callCuda(cudaFree(temp_data));
		callCuda(cudaFree(diff));
	}

	void EuclideanLoss::forward(bool train) {
		float a = 1;
		//callCuda(cudaMemcpy(&temp_data, prev->data, prev->data_size, cudaMemcpyDeviceToDevice));
		net_utils::setGpuValue(temp_data, prev->data_size, 0);
		// lable - prev->data 
		callCuda(cublasSaxpy(
			global::cublasHandle, 
			prev->data_size,
			&a,
			label, 1,
			temp_data, 1));
		a = -1;
		callCuda(cublasSaxpy(
			global::cublasHandle,
			prev->data_size,
			&a,
			prev->data, 1,
			temp_data, 1));

		callCuda(cublasSdot(
			global::cublasHandle,
			prev->data_size,
			temp_data,
			1,
			temp_data,
			1,
			data));
	}

	void EuclideanLoss::backward() {
		net_utils::setGpuValue(diff, prev->data_size, 0);
		// normalized by batch * data_size
		// ¶ÔÅ·ÊÏ¾àÀëÉèÖÃÈ¨ÖØËðÊ§
		//float a = -(loss_weight / (float)batch) / (float)prev->data_size;
		float a = -2.0 * loss_weight / (float)prev->data_size;
		callCuda(cublasSaxpy(
			global::cublasHandle,
			prev->data_size,
			&a,
			temp_data, 1,
			diff, 1));
	}

	void EuclideanLoss::update() {
		/*
		no parameters need upadting
		*/
	}

}