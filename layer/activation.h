#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include "layer.h"

namespace layer {

class Activation: public Layer {
private:
	cudnnActivationMode_t mode;
	cudnnActivationDescriptor_t activation_descriptor;
public:
	Activation(Layer* prev, cudnnActivationMode_t mode);
	virtual ~Activation();

	void forward(bool train = true);
	void backward();
	void update();
};

} /* namespace layer */
#endif /* ACTIVATION_H_ */
