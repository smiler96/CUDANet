#include "dec.h"

/*
* 功能：计算向量按分段长度累计求和, 同时按行累加而不是整个矩阵
* 如：输入维度为 KxHWC, 按block大小为C，reduce_size为HW，则输出向量大小为 KxC 
* 输入：vector_ptr	输入向量X的首地址
* 输入：vector_len	向量长度
* 输入：block_size	block的大小
* 输入：reduce_size	reduce的步长，为block_size整数倍

* 输出：sum_ptr	指向向量求和结果的指针
//*/
//__global__ void block_reduction_gpu(const float* __restrict__ vector_ptr, const int vector_len, const int block_size, const int reduce_size, float* __restrict__ sum_ptr)
//{
//	//<< < blockNum(K), block_size(blockDim.x C) >> > vector_len(K*HW*C) = blockNum(K) * reduce_size(HW*C), block_size (C)
//	//int index = threadIdx.x;
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	int vdex = blockIdx.x * reduce_size + threadIdx.x;
//	if (index < vector_len / reduce_size * block_size)
//	{
//		sum_ptr[index] = 0;
//		for (int i = 0; i < reduce_size / block_size; i++)
//		{
//			sum_ptr[index] += vector_ptr[vdex + i * block_size];
//		}
//		//for (int i = 0; i < reduce_size / block_size; i++)
//		//{
//		//	/*if(i==0) printf("(index %d): \n", index);
//		//	printf("%0.3f + ", vector_ptr[blockIdx.x * reduce_size + i * block_size + threadIdx.x]);
//		//	if (i == reduce_size / block_size-1) printf("\n", index);*/
//		//	printf("blockIdx.x-%d, threadIdx.x-%d, vdex-%d: %0.3f\n", blockIdx.x, threadIdx.x, index, blockDim.x, vdex + i * block_size, vector_ptr[vdex + i * block_size]);
//		//}
//	}
//}

/*
* 功能：构造函数，初始化
* 输入：featureNum			特征数
* 输入：centerNum			中心数
* 输入：featureDim			特征维度
*/
//DEC::DEC(const float* __restrict__ featureData_Dev, const float* __restrict__ centerData_Dev, int featureNum, int centerNum, int featureDim, bool diff)
DEC::DEC(int featureNum, int centerNum, int featureDim, bool diff)
{
	// 初始化参数
	ClusPara = clusterParam(featureNum, centerNum, featureDim);
	mClusPara = &ClusPara;
	needDiff = diff;

	// 初始化传入数据：图像特征数据，中心数据
	callCuda(cudaMalloc((void**)&mFeatureData_Dev, featureDim * featureNum * sizeof(float)));
	callCuda(cudaMalloc((void**)&mCenterData_Dev, featureDim * centerNum * sizeof(float)));
	/*callCuda(cudaMemcpy(mFeatureData_Dev, featureData_Dev, featureDim * featureNum * sizeof(float), cudaMemcpyDeviceToDevice));
	callCuda(cudaMemcpy(mCenterData_Dev, centerData_Dev, featureDim * centerNum * sizeof(float), cudaMemcpyDeviceToDevice));*/

	// 初始化前向内存分配
	callCuda(cudaMalloc((void**)&mResidual_Dev, featureNum * featureDim * centerNum * sizeof(float)));
	callCuda(cudaMalloc((void**)&mDistance_Dev, featureNum * centerNum * sizeof(float)));
	callCuda(cudaMalloc((void**)&mDistanceAdd1_Dev, featureNum * centerNum * sizeof(float)));
	callCuda(cudaMalloc((void**)&mDistanceAdd1Inv_Dev, featureNum * centerNum * sizeof(float)));
	callCuda(cudaMalloc((void**)&mDistanceAdd1InvSumj_Dev, featureNum * sizeof(float)));

	callCuda(cudaMalloc(&S2f_ptr, mClusPara->feaNum * mClusPara->cenNum * sizeof(float)));
	callCuda(cudaMalloc((void**)&sigma_S2fj_ptr, mClusPara->feaNum * sizeof(float)));

	callCuda(cudaMalloc((void**)&mSourceMetrix_Dev, featureNum * centerNum * sizeof(float)));
	callCuda(cudaMalloc((void**)&mSourceMetrix2_Dev, featureNum * centerNum * sizeof(float)));
	callCuda(cudaMalloc((void**)&mSoftCenFrequ_Dev,  centerNum * sizeof(float)));
	callCuda(cudaMalloc((void**)&mTargetMetrix_Dev, featureNum * centerNum * sizeof(float)));

	callCuda(cudaMalloc((void**)&mLossMetrix_Dev, featureNum * centerNum * sizeof(float)));

	cudaMalloc((void**)&_one_cenNum, mClusPara->cenNum * sizeof(float));
	cudaMalloc((void**)&_one_feaNum, mClusPara->feaNum * sizeof(float));
	cudaMalloc((void**)&_one, mClusPara->feaNum * mClusPara->cenNum * sizeof(float));
	
	// 初始化后向传播内存分配
	if (needDiff)
	{
		cudaMalloc((void**)&mTargetSubsSource_Dev, featureNum * centerNum * sizeof(float));
		cudaMalloc((void**)&mWeightDistri_Dev, featureNum * centerNum * sizeof(float));
		cudaMalloc((void**)&mWeightResidual_Dev, featureNum * featureDim * centerNum * sizeof(float));
		cudaMalloc((void**)&mLossToCenGrad_Dev, featureDim * centerNum * sizeof(float));
		
		cudaMalloc((void**)&mLossToFeaGrad_Dev, featureDim * featureNum * sizeof(float));
	}
}

/*
* 功能：计算聚类层前向传播损失
* 输入：featureData_Dev		特征矩阵
* 输入：centerData_Dev		聚类中心矩阵
* 输出：loss				前向传播损失
*/
void DEC::clusterLoss_gpu_forward()
{
	net_utils::setGpuValue(_one_cenNum, mClusPara->cenNum, 1);
	net_utils::setGpuValue(_one_feaNum, mClusPara->feaNum, 1);
	net_utils::setGpuValue(_one, mClusPara->feaNum * mClusPara->cenNum, 1);

	func_gpu::encodeResidualGpu(mClusPara->feaDim, mFeatureData_Dev, mClusPara->feaNum, mCenterData_Dev, mClusPara->cenNum, mResidual_Dev);
	/*std::cout << "func_gpu::encodeResidualGpu：\n";
	net_utils::printGpuMatrix(mResidual_Dev, mClusPara->feaDim * mClusPara->cenNum * mClusPara->feaNum, mClusPara->feaNum * mClusPara->cenNum, mClusPara->feaDim, 3);*/
	func_gpu::encodeDistanceGpu(mClusPara->feaDim, mFeatureData_Dev, mClusPara->feaNum, mCenterData_Dev, mClusPara->cenNum, mDistance_Dev);
	/*std::cout << "func_gpu::encodeDistanceGpu：\n";
	net_utils::printGpuMatrix(mDistance_Dev, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);*/

	func_gpu::processElementGpu(mDistance_Dev, mClusPara->feaNum * mClusPara->cenNum, 1, 1, mDistanceAdd1_Dev);
	func_gpu::inverseGpu(mDistanceAdd1_Dev, mClusPara->feaNum * mClusPara->cenNum, 1, mDistanceAdd1Inv_Dev);
	
	callCuda(cublasSgemv(global::cublasHandle, CUBLAS_OP_N, mClusPara->feaNum, mClusPara->cenNum, &alpha, mDistanceAdd1Inv_Dev, mClusPara->feaNum, 
		_one_cenNum, 1, &beta, mDistanceAdd1InvSumj_Dev, 1));
	/*std::cout << "cublasSgemv mDistanceAdd1InvSumj_Dev 1xHW ：\n";
	net_utils::printGpuMatrix(mDistanceAdd1InvSumj_Dev, mClusPara->feaNum, 1, mClusPara->feaNum, 3);*/

	func_gpu::divideElembyRowGpu(mDistanceAdd1Inv_Dev, mClusPara->cenNum * mClusPara->feaNum, mDistanceAdd1InvSumj_Dev, mClusPara->cenNum, 
		mClusPara->feaNum, mSourceMetrix_Dev);
	/*std::cout << "func_gpu::divideElembyRowGpu mSourceMetrix_Dev KxHW ：\n";
	net_utils::printGpuMatrix(mSourceMetrix_Dev, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);*/

	callCuda(cublasSgemv(global::cublasHandle, CUBLAS_OP_T, mClusPara->feaNum, mClusPara->cenNum, &alpha, mSourceMetrix_Dev, mClusPara->feaNum,
		_one_feaNum, 1, &beta, mSoftCenFrequ_Dev, 1));
	/*std::cout << "cublasSgemv mSoftCenFrequ_Dev 1xK ：\n";
	net_utils::printGpuMatrix(mSoftCenFrequ_Dev, mClusPara->cenNum, 1, mClusPara->cenNum, 3);*/

	func_gpu::powGpu(mSourceMetrix_Dev, mClusPara->feaNum *  mClusPara->cenNum, 2, 1, 0, mSourceMetrix2_Dev);

	/*std::cout << "mDistanceAdd1_Dev KxHW ：\n";
	net_utils::printGpuMatrix(mDistanceAdd1_Dev, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);
	std::cout << "mDistanceAdd1Inv_Dev KxHW ：\n";
	net_utils::printGpuMatrix(mDistanceAdd1Inv_Dev, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);
	std::cout << "mDistanceAdd1InvSumj_Dev 1xHW ：\n";
	net_utils::printGpuMatrix(mDistanceAdd1InvSumj_Dev, mClusPara->feaNum, 1, mClusPara->feaNum, 3);
	std::cout << "mSourceMetrix_Dev KxHW ：\n";
	net_utils::printGpuMatrix(mSourceMetrix_Dev, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);
	std::cout << "mSourceMetrix2_Dev KxHW ：\n";
	net_utils::printGpuMatrix(mSourceMetrix2_Dev, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);
	std::cout << "mSoftCenFrequ_Dev 1xK ：\n";
	net_utils::printGpuMatrix(mSoftCenFrequ_Dev, mClusPara->cenNum, 1, mClusPara->cenNum, 3);*/

	func_gpu::divideElembyColGpu(mSourceMetrix2_Dev, mClusPara->cenNum * mClusPara->feaNum, mSoftCenFrequ_Dev, mClusPara->cenNum, mClusPara->feaNum, S2f_ptr);
	/*std::cout << "func_gpu::divideElembyColGpu KxHW S2f_ptr ：\n";
	net_utils::printGpuMatrix(S2f_ptr, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);*/

	callCuda(cublasSgemv(global::cublasHandle, CUBLAS_OP_N, mClusPara->feaNum, mClusPara->cenNum, &alpha, S2f_ptr, mClusPara->feaNum, _one_cenNum, 1, &beta, sigma_S2fj_ptr, 1));
	/*std::cout << "cublasSgemv sigma_S2fj_ptr 1xHW ：\n";
	net_utils::printGpuMatrix(sigma_S2fj_ptr, mClusPara->feaNum, 1, mClusPara->feaNum, 3);*/
	
	func_gpu::divideElembyRowGpu(S2f_ptr, mClusPara->cenNum * mClusPara->feaNum, sigma_S2fj_ptr, mClusPara->cenNum, mClusPara->feaNum, mTargetMetrix_Dev);
	/*std::cout << "func_gpu::divideElembyRowGpu mTargetMetrix_Dev KxHW ：\n";
	net_utils::printGpuMatrix(mTargetMetrix_Dev, mClusPara->feaNum * mClusPara->cenNum, mClusPara->cenNum, mClusPara->feaNum, 3);*/
	
	func_gpu::divergKLGpu(mTargetMetrix_Dev, mSourceMetrix_Dev, mClusPara->feaNum * mClusPara->cenNum, mLossMetrix_Dev);

	callCuda(cublasSdot(global::cublasHandle, mClusPara->feaNum * mClusPara->cenNum, mLossMetrix_Dev, 1, _one, 1, &Loss));
	checkValidNum(Loss);

	/*std::cout << "S2f_ptr KxHW ：\n";
	net_utils::printGpuMatrix(S2f_ptr, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);
	std::cout << "sigma_S2fj_ptr 1xK ：\n";
	net_utils::printGpuMatrix(sigma_S2fj_ptr, mClusPara->feaNum, 1, mClusPara->feaNum, 3);
	std::cout << "mTargetMetrix_Dev KxHW ：\n";
	net_utils::printGpuMatrix(mTargetMetrix_Dev, mClusPara->feaNum * mClusPara->cenNum, mClusPara->cenNum, mClusPara->feaNum, 3);
	std::cout << "mLossMetrix_Dev KxHW ：\n";
	net_utils::printGpuMatrix(mLossMetrix_Dev, mClusPara->feaNum * mClusPara->cenNum, mClusPara->cenNum, mClusPara->feaNum, 3);*/
	//std::cout << "Loss：" << Loss << "\n";
}

/*
* 功能：计算聚类层后向传播梯度
* 输入：featureData_Dev		特征矩阵
* 输入：centerData_Dev		聚类中心矩阵
* 输出：loss				前向传播损失
*/
void DEC::clusterLoss_gpu_backward()
{
	if (needDiff)
	{
		//std::cout << "clusterLoss_gpu_backward：\n";
		net_utils::setGpuValue(_one_cenNum, mClusPara->cenNum, 1);
		net_utils::setGpuValue(_one_feaNum, mClusPara->feaNum, 1);

		net_utils::setGpuValue(mLossToCenGrad_Dev, mClusPara->feaDim * mClusPara->cenNum, 0);
		net_utils::setGpuValue(mLossToFeaGrad_Dev, mClusPara->feaDim * mClusPara->feaNum, 0);

		func_gpu::vectorAddGpu(mClusPara->feaNum * mClusPara->cenNum, mTargetMetrix_Dev, 1, mSourceMetrix_Dev, -1, 0, mTargetSubsSource_Dev);
		/*std::cout << "mTargetSubsSource_Dev KxHW ：\n";
		net_utils::printGpuMatrix(mTargetSubsSource_Dev, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);*/

		func_gpu::vectordivideElemGpu(mTargetSubsSource_Dev, mDistanceAdd1_Dev, mClusPara->feaNum * mClusPara->cenNum, 1, 0, mWeightDistri_Dev);
		/*std::cout << "mWeightDistri_Dev KxHW ：\n";
		net_utils::printGpuMatrix(mWeightDistri_Dev, mClusPara->cenNum * mClusPara->feaNum, mClusPara->cenNum, mClusPara->feaNum, 3);
		std::cout << "mResidual_Dev：\n";
		net_utils::printGpuMatrix(mResidual_Dev, mClusPara->feaDim * mClusPara->cenNum * mClusPara->feaNum, mClusPara->feaNum * mClusPara->cenNum, mClusPara->feaDim, 3);*/

		func_gpu::asyMultiplyElemGpu(mResidual_Dev, mClusPara->feaNum * mClusPara->cenNum * mClusPara->feaDim, mWeightDistri_Dev, mClusPara->feaNum * mClusPara->cenNum, 
			mClusPara->feaDim, mWeightResidual_Dev);
		/*std::cout << "func_gpu::asyMultiplyElemGpu mWeightResidual_Dev 1xKxHWxC ：\n";
		net_utils::printGpuMatrix(mWeightResidual_Dev, mClusPara->feaDim * mClusPara->cenNum * mClusPara->feaNum, mClusPara->feaNum * mClusPara->cenNum, mClusPara->feaDim, 3);*/

		/*float* _one_cenNum = nullptr;
		cudaMalloc((void**)&_one_cenNum, mClusPara->cenNum * sizeof(float));*/
		net_utils::setGpuValue(_one_cenNum, mClusPara->cenNum, 1);
		callCuda(cublasSgemv(global::cublasHandle, CUBLAS_OP_N, mClusPara->feaNum * mClusPara->feaDim, mClusPara->cenNum,
			&alpha, mWeightResidual_Dev, mClusPara->feaNum * mClusPara->feaDim, _one_cenNum, 1, &beta, mLossToFeaGrad_Dev, 1));
		/*std::cout << "cublasSgemv mLossToFeaGrad_Dev 1xHWxC ：\n";
		net_utils::printGpuMatrix(mLossToFeaGrad_Dev, mClusPara->feaDim * mClusPara->feaNum, mClusPara->feaNum, mClusPara->feaDim, 3);*/

		/*block_reduction_gpu << < mClusPara->cenNum, mClusPara->feaDim >> > (mWeightResidual_Dev, mClusPara->feaNum * mClusPara->cenNum * mClusPara->feaDim, mClusPara->feaDim, 
			mClusPara->feaNum * mClusPara->feaDim, mLossToCenGrad_Dev);
		std::cout << "mLossToCenGrad_Dev 1xKxC ：\n";
		net_utils::printGpuMatrix(mLossToCenGrad_Dev, mClusPara->feaDim * mClusPara->cenNum, mClusPara->cenNum, mClusPara->feaDim, 3);*/

		//net_utils::setGpuValue(_one_feaNum, mClusPara->feaNum, 1);
		//net_utils::setGpuValue(mLossToCenGrad_Dev, mClusPara->feaDim * mClusPara->cenNum, 0);
		for (int k = 0; k < mClusPara->cenNum; k++)
		{
			callCuda(cublasSgemv(global::cublasHandle, CUBLAS_OP_N, mClusPara->feaDim, mClusPara->feaNum, &alpha, mWeightResidual_Dev + k * mClusPara->feaNum * mClusPara->feaDim, mClusPara->feaDim,
				_one_feaNum, 1, &beta, mLossToCenGrad_Dev + k * mClusPara->feaDim, 1));

			//net_utils::printGpuMatrix(mLossToCenGrad_Dev + k * mClusPara->feaDim, mClusPara->feaDim, 1, mClusPara->feaDim, 3);
		}
		/*std::cout << "cublasSgemv mLossToCenGrad_Dev 1xKxC ：\n";
		net_utils::printGpuMatrix(mLossToCenGrad_Dev, mClusPara->feaDim * mClusPara->cenNum, mClusPara->cenNum, mClusPara->feaDim, 3);*/
	}
}

void DEC::input_data(const float* feaData_dev, const float* cenData_dev)
{
	callCuda(cudaMemcpy(mFeatureData_Dev, feaData_dev, mClusPara->feaDim * mClusPara->feaNum * sizeof(float), cudaMemcpyDeviceToDevice));
	callCuda(cudaMemcpy(mCenterData_Dev, cenData_dev, mClusPara->feaDim * mClusPara->cenNum * sizeof(float), cudaMemcpyDeviceToDevice));
}

/*
* 功能：析构函数
*/
DEC::~DEC()
{
	// 释放内存
	callCuda(cudaFree(mFeatureData_Dev));
	callCuda(cudaFree(mCenterData_Dev));

	callCuda(cudaFree(mResidual_Dev));
	callCuda(cudaFree(mDistance_Dev));
	callCuda(cudaFree(mSourceMetrix_Dev));
	callCuda(cudaFree(mTargetMetrix_Dev));
	callCuda(cudaFree(mLossMetrix_Dev));

	callCuda(cudaFree(_one_cenNum));
	callCuda(cudaFree(_one_feaNum));
	callCuda(cudaFree(_one));
	callCuda(cudaFree(S2f_ptr));
	callCuda(cudaFree(sigma_S2fj_ptr));

	// 释放导数内存
	if (needDiff)
	{
		callCuda(cudaFree(mWeightDistri_Dev));
		callCuda(cudaFree(mWeightResidual_Dev));
		callCuda(cudaFree(mLossToCenGrad_Dev));
		callCuda(cudaFree(mLossToFeaGrad_Dev));
	}
}

