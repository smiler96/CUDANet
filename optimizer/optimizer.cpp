#include "optimizer.h"


namespace optimizer
{

	Optimizer::Optimizer(float _learn_rate, float _weight_decay)
	{
		learn_rate = _learn_rate;
		weight_decay = _weight_decay;

		inv_weight_decay = 1 - weight_decay;
	}


	Optimizer::~Optimizer()
	{
	}
}