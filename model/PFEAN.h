/*
* PFEAN.h
*
* prior guided feature encodere adversarial network
*
*  Created on: 1 11, 2020
*      Author: wanqian
*/

#ifndef _PFEAN_H_
#define _PFEAN_H_

#include "gan.h"
#include "../utils/image_priors.h"

#define PRIOR_NUM 3

namespace PFEAN
{
	using namespace net_utils;
	using namespace model;

	class PriorExtractor : public Prior
	{
	public:
		PriorExtractor() : Prior() {};
		~PriorExtractor() {};
	};


	class PFEANet : public GAN 
	{
	public:

		PFEANet(float* _h_data, int _h, int _w, int _c, int _batch, int _data_size);

		~PFEANet();

	public:
		int size; // 训练数据的样本个数
		int batch; // batch大小
		int data_dim; // 输入数据维度 HxWxC
		float* h_data; // host 输入图像数据
		float* d_data; // device 传入数据， 包括原图和先验提取信息 Bx256x256x4
		float* d_label; // device 传入数据标签
	};
}

#endif // !_PFEAN_H_
