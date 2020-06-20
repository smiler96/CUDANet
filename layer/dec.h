#pragma once

#ifndef DEC_H_
#define DEC_H_

#include <cublas_v2.h>
#include "../utils/global.h"
#include "../utils/set_value.h"

#include "../utils/utils.h"
#include "../utils/common_funcs_gpu.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <stdlib.h>

struct clusterParam
{
	int feaNum = 0; // 特征数
	int cenNum = 0;  // 中心数
	int feaDim = 0; // 特征维度

	clusterParam(int x, int y, int c) :feaNum(x), cenNum(y), feaDim(c) {};
};

class DEC
{
public:
	clusterParam ClusPara = clusterParam(0, 0, 0);
	clusterParam *mClusPara;
	bool needDiff;

	float* mFeatureData_Dev; // 图像编码特征，ci, 维度： 1 C HW
	float* mCenterData_Dev; // 聚类中心，μj, 维度： 1 C K

	// 前向传播数据定义
	/*float* mDistanceMetrix_Dev;*/
	float* mResidual_Dev; // 图像编码特征到编码中心的残差，ci-μj, 维度： 1 HW C K
	float* mDistance_Dev; // 图像编码特征到编码中心的残差距离，即，编码残差 ||ci-μj||^2, 维度： 1 HW K
	float* mDistanceAdd1_Dev; // 图像编码特征到编码中心的残差距离加1，即，编码残差 1+||ci-μj||^2, 维度： 1 HW K
	float* mDistanceAdd1Inv_Dev; // 图像编码特征到编码中心的残差距离加1的倒数，即，编码残差 (1+||ci-μj||^2)^-1, 维度： 1 HW K
	float* mDistanceAdd1InvSumj_Dev; // 图像编码特征到编码中心的残差距离加1的倒数对μj'求和，即，编码残差 Σj'(1+||ci-μj'||^2)^-1, 维度： 1 HW

	float* S2f_ptr = nullptr; // Sij^2/fj 维度为 1 HW K
	float* sigma_S2fj_ptr = nullptr; // Σj Sij^2/fj  维度为 1 K

	float* mSourceMetrix_Dev; // 图像编码特征到编码中心的源域分布，Sij = (1+||ci-μj||^2)^-1 / Σj((1+||ci-μj||)^-1), 维度： 1 HW K
	float* mSourceMetrix2_Dev; // 图像编码特征到编码中心的源域分布平方，Sij^2, 维度： 1 HW K
	float* mSoftCenFrequ_Dev; // 图像编码特征到编码中心的软聚类频率，fj = Σi(Sij), 维度： 1 K
	float* mTargetMetrix_Dev; // 图像编码特征到编码中心的目标分布，Tij = Sij^2/fj / Σj(Sij^2/fj), 维度： 1 HW K

	float* mLossMetrix_Dev; // 图像聚类损失，Lij = ΣiΣj(Tij*log(Tij/Sij)), 维度： 1 HW K
	float Loss=0; // 图像聚类损失求和，Loss = ΣiΣj(Lij), 维度： 1 

	float* _one_cenNum = nullptr;
	float* _one_feaNum = nullptr;
	float* _one = nullptr;

	// 反向传播梯度数据定义
	float alpha = 1.0, beta = 0;

	float* mTargetSubsSource_Dev; // 目标分布减去源域分布，Tij - Sij, 维度： 1 HW K
	float* mWeightDistri_Dev; // (1+||ci-μj||^2)^-1 * (Tij - Sij), 维度： 1 HW K 
	float* mWeightResidual_Dev; // (1+||ci-μj||^2)^-1 * (Tij - Sij) *（ci-μj）, 维度： 1 HW C K 
	float* mLossToCenGrad_Dev; //  图像聚类损失Loss对中心Cen的导数，∂Loss/∂ci = ∂Loss/∂S * ∂S/∂ci + ∂Loss/∂T * ∂T/∂S * ∂S/∂ci, 维度： 1 C K
	float* mLossToFeaGrad_Dev; //  图像聚类损失Loss对特征Fea的导数，∂Loss/∂μj = ∂Loss/∂S * ∂S/∂μj + ∂Loss/∂T * ∂T/∂S * ∂S/∂μj, 维度： 1 C HW
	
public:
	DEC(int featureNum, int centerNum, int featureDim, bool diff=false);
	~DEC();

	void input_data(const float* feaData_dev, const float* cenData_dev);

	void clusterLoss_gpu_forward();
	void clusterLoss_gpu_backward();
};

#endif // DEC_H_