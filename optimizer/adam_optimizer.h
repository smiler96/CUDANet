#pragma once
#ifndef _ADAM_H_
#define _ADAM_H_

#include "optimizer.h"
#include "../utils/common_funcs_gpu.h"

namespace optimizer
{
	class Adam : public Optimizer
	{
	public:
		Adam(float _learn_rate, float _weight_decay, float _beta1, float _beta2, float _epsilon);
		virtual ~Adam();

		float beta1;
		float beta2;
		float epsilon;
		float invbeta1; // 1-beta1
		float invbeta2; // 1-beta2
		float m1_hat; // 1/(1 - beta1^t)
		float m2_hat; // 1/(1 - beta2^t)

		void adjust_lr(float _step_decay);
		void optimize(Layer* layer, const int t);

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

#endif // !_ADAM_H_

