/*
class classification inherits from Network
for classification model
20200103
created by wanqian
*/

#pragma once
#ifndef _GAN_H_
#define _GAN_H_

#include "../utils/image.h"
#include "network.h"
#include "reconstruction.h"
#include "../optimizer/adam_optimizer.h"
#include "../optimizer/sgd_optimizer.h"

namespace model
{
	class GAN
	{
	public:
		GAN() {};
		GAN(float* _data, int _data_dim, int _train_size, int _batch);

		int batch;
		int train_size;		// 训练图片数
		int data_dim;		// 单个数据维数 H*W*C
		int label_dim;		// 单个标签维数（此处同输入数据） H*W*C(重构)

		// cpu
		float* h_data;			// 所有原始图像数据
		float* reconstruction_data;
		float* dis_label_h1;
		float* dis_label_h0;
		// gpu
		float* data;			// 一个batch的原始数据
		float* rec_data;		// 生成器重构数据
		float* rec_label;		// 原始重构数据标签
		//float* dis_label;		// 分类标签
		float* ONE;				// 1标签
		float* ZERO;			// 0标签

		float* hybrid_data;		// 判别器训练时的混合数据
		float* hybrid_label;	// 判别器训练时的混合标签

		Reconstruction* generator;
		Network* discrimintor;

	private:
		// cpu
		float dis_loss_1;
		float dis_loss_0;

		float gen_loss_rec;
		float gen_loss_1;
		// gpu
		float* dis_diff_1;			// 长度为 softmax 输出数据的长度，即为 batch*class_num(2)
		float* dis_diff_0;

		float* gen_diff_rec;		// 生成器重构数据导数
		float* gen_diff_1;			// 判别器对生成数据反向传播到生成数据处的导数，二者的长度均为重构数据的长度


	public:
		~GAN();

		void TrainDis(optimizer::SGD sgd, optimizer::Adam adam, int adam_iter);
		void TrainGen(optimizer::SGD sgd, optimizer::Adam adam, int adam_iter);

		void Train(std::string para_path, int iteration, float step_decrease, bool debug);
		void Test();

		void SaveParas(std::string _para_path);
		void ReadGenParas(std::string _para_path);
		void ReadDisParas(std::string _para_path);
		void ReadParas(std::string _para_path);

	};

}

#endif // !_GAN_H_
