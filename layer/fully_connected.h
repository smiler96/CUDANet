#ifndef FC_H_
#define FC_H_

#include "layer.h"

namespace layer {

	// fully connected layer
	class FullyConnected : public Layer {

	private:
		float dropout_rate;

	protected:
		int input_size; // output size of previous layer
		int output_size; // output size
		float* one; // full one vector for bias
		void dropout(bool train);

	public:
		// data is output_size * batch
		FullyConnected(Layer* _prev, int _output_size, float dropout_rate, float alpha,
			float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);
		virtual ~FullyConnected();

		void forward(bool train = true);
		void backward();
		void update();
	};

} /* namespace layer */
#endif /* FC_H_ */
