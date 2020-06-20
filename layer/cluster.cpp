#include "cluster.h"

// 接在卷积层后
namespace layer {

	Cluster::Cluster(Layer* _prev, int K, float cluster_weight, float alpha, float* init_param,
		float sigma, float momentum, float weight_decay) :
		Layer(alpha, momentum, weight_decay) {
		prev = _prev;
		prev->next = this;
		
		cenNum = K;
		weight = cluster_weight;

		//直接输出输入的前一层的数据
		t_data = prev->t_data;
		data_size = prev->data_size;
		data = prev->data;

		cudnnDataType_t _t;
		int _n, _c, _h, _w, _tmp;
		callCudnn(cudnnGetTensor4dDescriptor(prev->t_data, &_t, &_n, &_c, &_h, &_w, &_tmp, &_tmp, &_tmp, &_tmp));

		feaNum = _h * _w;
		feaDim = _c;
		batch = _n;
		dataDim = prev->data_size / batch;

		callCudnn(cudnnCreateTensorDescriptor(&fea_discriptor));
		callCudnn(cudnnSetTensor4dDescriptor(fea_discriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, _n, _c, _h, _w));

		param_size = _c * cenNum; //聚类中心的参数个数
		param_bias_size = 0;
		callCuda(cudaMalloc(&param, sizeof(float) * param_size));
		callCuda(cudaMalloc(&gradient, sizeof(float) * param_size));
		callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size)); // 梯度回传需要的数据空间
		callCuda(cudaMalloc(&fea_data, sizeof(float) * prev->data_size)); // 梯度回传需要的数据空间
		callCuda(cudaMalloc(&fea_diff, sizeof(float) * prev->data_size)); // 梯度回传需要的数据空间

		dec = new DEC(feaNum, cenNum, feaDim, true);

		if (init_param == NULL)
			net_utils::setGpuNormalValue(param, param_size, 0, sigma);
		else
			callCuda(cudaMemcpy(param, init_param, sizeof(float) * param_size, cudaMemcpyHostToDevice));
	}

	Cluster::~Cluster() {
		callCudnn(cudnnDestroyTensorDescriptor(t_data));
		callCuda(cudaFree(data));
		callCuda(cudaFree(diff));
		callCuda(cudaFree(param));
		callCuda(cudaFree(gradient));

		delete dec;
	}

	void Cluster::forward(bool train) 
	{
		float a = 1.0, b = 0;
		callCudnn(cudnnTransformTensor(cudnnHandle, &a, t_data, data, &b, fea_discriptor, fea_data));
	}

	void Cluster::backward() 
	{
		// 初始化损失
		cluster_loss = 0;
		net_utils::setGpuValue(gradient, param_size, 0);
		net_utils::setGpuValue(diff, data_size, 0);
		net_utils::setGpuValue(fea_diff, data_size, 0);

		// 分配单张图像内存
		float* dataSingle;
		callCuda(cudaMalloc(&dataSingle, sizeof(float) * data_size / batch));

		int offset = 0;
		float nrm = 1.0 / batch;
		// 1.逐张图片计算聚类损失 KL （前向传播）
		for (int i = 0; i < batch; i++)
		{
			callCuda(cudaMemcpy(dataSingle, fea_data + offset * dataDim, sizeof(float) * dataDim, cudaMemcpyDeviceToDevice));

			dec->input_data(dataSingle, param);

			dec->clusterLoss_gpu_forward();
			cluster_loss += dec->Loss;

			dec->clusterLoss_gpu_backward();
			callCuda(cudaMemcpy(fea_diff + offset * dataDim, dec->mLossToFeaGrad_Dev, sizeof(float) * dataDim, cudaMemcpyDeviceToDevice));
			callCuda(cublasSaxpy(cublasHandle, param_size, &nrm, dec->mLossToCenGrad_Dev, 1, gradient, 1));

			offset += batch;
		}

		float a = 1.0, b = 0;
		// NHWC to NCHW
		callCudnn(cudnnTransformTensor(cudnnHandle, &a, fea_discriptor, fea_diff, &b, t_data, diff));
		callCuda(cublasSaxpy(cublasHandle, data_size, &weight, next->diff, 1, diff, 1));
	}

	void Cluster::update() 
	{
		float a = alpha;
		// update the convolution filter parameters
		callCuda(cublasSaxpy(cublasHandle, param_size, &a, gradient, 1, param, 1));
	}

}
