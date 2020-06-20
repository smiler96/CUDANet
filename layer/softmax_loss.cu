#include "softmax_loss.h"

using namespace global;

namespace layer {

	SoftmaxLoss::SoftmaxLoss(Layer* _prev, int _class_num, float dropout_rate, float alpha,
		float sigma, float momentum, float weight_decay) :
		Neuron(_prev, _class_num, dropout_rate, alpha, sigma) {
		prev = _prev;
		prev->next = this;

		//label = _label; // gpu datas
		class_num = _class_num;
		//batch = _batch;
		data_size = batch;
		param_size = 0;
		param_bias_size = 0;

		callCudnn(cudnnCreateTensorDescriptor(&t_data));
		callCudnn(cudnnSetTensor4dDescriptor(
			t_data,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batch, 1, 1, 1));

		//callCuda(cudaMalloc(&data, sizeof(float) * data_size));
		callCuda(cudaMalloc(&softmaxdata, sizeof(float) * prev->data_size));  
		callCuda(cudaMalloc(&loss, sizeof(float) * 1));
		callCuda(cudaMalloc(&data, sizeof(float) * data_size)); // 此处用于存放 loss
		callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size));
	}

	SoftmaxLoss::~SoftmaxLoss() {
		callCudnn(cudnnDestroyTensorDescriptor(t_data));
		callCuda(cudaFree(softmaxdata));
		callCuda(cudaFree(loss));
		callCuda(cudaFree(data));
		callCuda(cudaFree(label));
		callCuda(cudaFree(diff));
	}

	__global__ void _predict(const float *softmax, int label_dim, int batch, float *data)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= batch)
			return;

		int label_value = 0;
		float max = -1;
		for (int i = 0; i < label_dim; i++) {
			if (softmax[idx * label_dim + i] > max) {
				max = softmax[idx * label_dim + i];
				label_value = i;
			}
		}

		data[idx] = (float)label_value;
	}

	__global__ void _softmaxloss(const float *softmax, float *label, int class_num, int batch, float *data)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < batch)
		{
			atomicAdd(data, -log(softmax[idx * class_num + (int)label[idx]]));
		}
	}

	__global__ void _softmaxDiff(const float *label, const float *softmaxVal, int class_num, int batch, float *diff)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < batch)
		{
			const int label_value = static_cast<int>(label[idx]);
			diff[idx * class_num + label_value] -= 1.0f; // 对zi求导
			//diff[idx * class_num + label_value] = -1.0f / softmaxVal[idx * class_num + label_value]; // 对ei求导
		}
	}

	void SoftmaxLoss::forward(bool train) {
		//net_utils::printGpuMatrix(prev->data, prev->data_size, prev->data_size);

		float a = 1;
		float b = 0;
		callCudnn(cudnnSoftmaxForward(cudnnHandle,
			CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			&a,
			prev->t_data, prev->data,
			&b,
			prev->t_data, softmaxdata));

		net_utils::setGpuValue(loss, 1, 0);

		// 计算 softmax loss
		_softmaxloss <<< (batch + 127) / 128, 128 >>> (softmaxdata, label, class_num, batch, loss);

		// 预测标签
		_predict << < (batch + 127) / 128, 128 >> > (softmaxdata, class_num, batch, data);
	}

	void SoftmaxLoss::backward() {
		callCuda(cudaMemcpy(diff, softmaxdata, sizeof(float) * prev->data_size, cudaMemcpyDeviceToDevice));
		_softmaxDiff <<< (batch + 127) / 128, 128 >>> (label, softmaxdata, class_num, batch, diff);
	}

	void SoftmaxLoss::update() {
		// nothing
	}

}
