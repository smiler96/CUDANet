#pragma once

#ifndef _COMMON_FUNCS_GPU_
#define _COMMON_FUNCS_GPU_

#include <ctime>
#include <cmath>
#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"

#include "utils.h"
#include "global.h"
#include "print.h"

#define ThreadsPerBlock_16 16
#define ThreadsPerBlock_32 32

namespace func_gpu {

	/*
	* 功能：计算向量的倒数 y[i] = aplha / x[i]
	* 输入：x	向量的首地址
	* 输入：n	向量长度
	* 输入：aplha	乘系数

	* 输出：y	向量R的首地址
	*/
	void inverseGpu(const float* __restrict__ x, int n, float alpha, float* __restrict__ y);

	/*
	* 功能：计算向量幂 y[i] = aplha*x[i]^p + b
	* 输入：x	向量X的首地址
	* 输入：n	向量长度
	* 输入：p	幂因子
	* 输入：aplha	乘系数
	* 输入：b	加系数

	* 输出：y	指向输出向量结果的指针
	*/
	void powGpu(const float* __restrict__  x, int n, int p, float alpha, float b, float* __restrict__ y);

	/*
	* 功能：计算向量log y[i] = aplha*log(x[i]) + b
	* 输入：x	向量X的首地址
	* 输入：n	向量长度
	* 输入：p	幂因子
	* 输入：aplha	乘系数
	* 输入：b	加系数

	* 输出：y	指向输出向量结果的指针
	*/
	void logGpu(const float* __restrict__  x, int n, float alpha, float b, float* __restrict__  y);

	/*
	* 功能：计算向量与单个数 ponitwise 操作 y[i] = x[i]*alpha + b; 
	* 输入：x	向量的首地址
	* 输入：n	向量长度
	* 输入：aplha	乘子
	* 输入：b	被加数

	* 输出：y	向量R的首地址
	*/
	void processElementGpu(const float* __restrict__  x, int n, float alpha, float b, float* __restrict__  y);

	/*
	* 功能：计算两个向量 z[i] = a*x[i] + b*y[i] + c
	* 输入：n	向量长度
	* 输入：x	向量x的首地址
	* 输入：a	x乘子
	* 输入：y	向量y的首地址
	* 输入：b	y乘子
	* 输入：c	加系数

	* 输出：z	输出向量z的首地址
	*/
	void vectorAddGpu(int n, const float* __restrict__ x, float a, const float* __restrict__ y, float b, float c, float* __restrict__ z);

	/*
	* 功能：计算向量与向量元素相除 z[i] = alpha*x[i]/y[i] + b;
	* 输入：x	向量x的首地址
	* 输入：y	向量y的首地址
	* 输入：n	向量长度
	* 输入：aplha	乘系数
	* 输入：b	加系数

	* 输出：z	向量z的首地址
	*/
	void vectordivideElemGpu(const float* __restrict__ x, const float* __restrict__ y, int n, float alpha, float b, float* __restrict__ z);

	/*
	* 功能：计算向量与向量元素相乘 z[i] = alpha*x[i]*y[i] + b;
	* 输入：x	向量x的首地址
	* 输入：y	向量y的首地址
	* 输入：n	向量长度
	* 输入：aplha	乘系数
	* 输入：b	加系数

	* 输出：z	向量z的首地址
	*/
	void vectorMultiplyElemGpu(const float* __restrict__ x, const float* __restrict__ y, int n, float alpha, float b, float* __restrict__ z);

	/*
	* 功能：计算向量按行对应元素相除 z[i][j] = x[i][j] / y[i]
	* 输入：x	向量x的首地址		维度 n = rows*cols
	* 输入：n	向量x长度
	* 输入：y	向量y的首地址		维度 1*cols
	* 输入：rows		向量x行数
	* 输入：clos		向量x列数

	* 输出：z	向量z的首地址		维度 n = rows*cols
	*/
	void  divideElembyRowGpu(const float* __restrict__ x, int n, const float* __restrict__ y, int rows, int clos, float* __restrict__ z);

	/*
	* 功能：计算向量按列对应元素相除 z[i][j] = x[i][j] / y[j]
	* 输入：x	向量x的首地址		维度 n = rows*cols
	* 输入：n	向量x长度
	* 输入：y	向量y的首地址		维度 rows*1
	* 输入：rows		向量x行数
	* 输入：clos		向量x列数

	* 输出：z	向量z的首地址		维度 n = rows*cols
	*/
	void  divideElembyColGpu(const float* __restrict__ x, int n, const float* __restrict__ y, int rows, int clos, float* __restrict__ z);


	/*
	* 功能：对向量x采用 不等长向量y加权	z[i] = x[i] * y[i/step]
	* 输入：x	向量A的首地址		维度 (rows*step) * cols
	* 输入：xN	向量x长度
	* 输入：y	向量y的首地址	维度 rows * cols
	* 输入：yN	向量y长度

	* 输出：z	向量z的首地址		维度 n = rows*cols
	*/
	void asyMultiplyElemGpu(const float* __restrict__ x, int xN, const float* __restrict__ y, int yN, int step, float* z);

	/*
	* 功能：计算向量feature与向量center编码编码残差residual
	* 输入：fDim		feature和center的特征维度
	* 输入：feature	向量feature的首地址 维度 fDim * feaNum
	* 输入：feaNum	feature特征的个数
	* 输入：center	向量center的首地址 维度 fDim * CenNum
	* 输入：cenNum	center特征的个数

	* 输出：residual		编码残差residual的首地址 维度  fDim * feaNum * CenNum
	*/
	void encodeResidualGpu(int fDim, const float* __restrict__ feature, int feaNum, const float* __restrict__ center, int cenNum, float* __restrict__ residual);

	/*
	* 功能：计算向量feature与向量center编码距离distance
	* 输入：fDim		feature和center的特征维度
	* 输入：feature	向量feature的首地址 维度 fDim * feaNum
	* 输入：feaNum	feature特征的个数
	* 输入：center	向量center的首地址 维度 fDim * CenNum
	* 输入：cenNum	center特征的个数

	* 输出：distance		编码距离distance的首地址 维度  feaNum * CenNum
	*/
	void encodeDistanceGpu(int fDim, const float* __restrict__ feature, int feaNum, const float* __restrict__ center, int cenNum, float* __restrict__ distance);

	/*
	* 功能：计算向量x与向量y的KL散度矩阵 z[i] = x[i] * log(x[i] / y[i]);
	* 输入：x	向量x的首地址
	* 输入：y	向量y的首地址
	* 输入：n	向量长度

	* 输出：z	向量z的首地址
	*/
	void divergKLGpu(const float* __restrict__ x, const float* __restrict__ y, int n, float* z);

	/*
	* 功能：adam优化器梯度更新部分 gradient_update[i] = m1[i]/(1-β1^t) / sqrt(m2[i]/(1-β2^t) + epsilon) ;
	* 输入：m1	一阶动量的首地址
	* 输入：m2	二阶动量的首地址
	* 输入：n	向量长度
	* 输入：iter		迭代次数
	* 输入：beta1	adam优化系数β1 0.99
	* 输入：beta2	adam优化系数β2 0.999
	* 输入：epsilon	防止分母为0

	* 输出：gradient_update	梯度更新的首地址
	*/
	void adamUpdateGpu(const float* m1, const float* m2, const int n, const int iter, const float beta1, const float beta2, const float epsilon, float* gradient_update);
}

#endif