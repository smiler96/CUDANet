#pragma once

#ifndef SOFTMAXLOSS_CUH_
#define SOFTMAXLOSS_CUH_

#include "neuron.h"


namespace layer {

	class SoftmaxLoss : public Neuron {
	public:
		float* label;	// real label
		float* loss;	// real label
		float* softmaxdata;
		int class_num;	// eg. 10 for digit classfication

	public:
		SoftmaxLoss(Layer* _prev, int _output_size, float dropout_rate, float alpha,
			float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);
		virtual ~SoftmaxLoss();

		void forward(bool train = true);
		void backward();
		void update();
	};

} /* namespace layer */
#endif /* SOFTMAXLOSS_CUH_ */