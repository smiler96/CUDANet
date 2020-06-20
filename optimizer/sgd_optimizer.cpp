#include "sgd_optimizer.h"

namespace optimizer
{
	SGD::SGD(float _learn_rate, float _weight_decay) :
		Optimizer(_learn_rate, _weight_decay)
	{
	}

	SGD::~SGD()
	{

	}

	void SGD::optimize(Layer *layer)
	{
		//std::cout << "optimize" << " - layer:" << layer->getLayerName() << "_" << layer->getLayerId() << "\n";

		/*if (layer->next != nullptr)
		{
			net_utils::checkArrayNan(layer->next->diff, layer->data_size);
		}*/
		// update the convolution filter parameters
		if (layer->param_size > 0)
		{
			//net_utils::checkArrayNan(layer->gradient, layer->param_size);
			callCuda(cublasSscal(global::cublasHandle, layer->param_size, &inv_weight_decay, layer->param, 1));
			callCuda(cublasSaxpy(global::cublasHandle, layer->param_size, &learn_rate, layer->gradient, 1, layer->param, 1));
		}

		if (layer->param_bias_size > 0)
		{
			//net_utils::checkArrayNan(layer->gradient_bias, layer->param_bias_size);
			callCuda(cublasSscal(global::cublasHandle, layer->param_bias_size, &inv_weight_decay, layer->param_bias, 1));
			callCuda(cublasSaxpy(global::cublasHandle, layer->param_bias_size, &learn_rate, layer->gradient_bias, 1, layer->param_bias, 1));
		}
	}

}