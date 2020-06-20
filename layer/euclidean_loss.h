#ifndef _H_EUCLIDEAN_LOSS_H_
#define _H_EUCLIDEAN_LOSS_H_

#include "layer.h"

namespace layer {

	class EuclideanLoss : public Layer {
	public:
		float* label; // real label (always train data for reconstruction task)
		float* temp_data;
		float loss;
		float loss_weight;
	public:
		EuclideanLoss(Layer* _prev, float* _label, float _loss_weight);
		virtual ~EuclideanLoss();
		void forward(bool train = true);
		void backward();
		void update();
	};

}
#endif // !_H_EUCLIDEAN_LOSS_H_
