#pragma once
#ifndef _SGD_H_
#define _SGD_H_

#include "optimizer.h"
#include "../utils/common_funcs_gpu.h"

namespace optimizer
{
	class SGD : public Optimizer
	{
	public:
		SGD(float _learn_rate, float _weight_decay);
		virtual ~SGD();

		void optimize(Layer *layer);

		//float* para_moment1; // first moment vector
		//float* para_moment2; // second moment vector
		//float* para_m1_hat;
		//float* para_m2_hat;
		//float* bias_moment1; // first moment vector
		//float* bias_moment2; // second moment vector
		//float* bias_m1_hat;
		//float* bias_m2_hat;

	};
}


#endif // !_SGD_H_
