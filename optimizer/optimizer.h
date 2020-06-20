/*
* optimizer.h
*
* ÓÅ»¯Æ÷
* 20191224, by wanqian
*
*/
#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#define OPTIMIZER_SGD 0
#define OPTIMIZER_ADAM 1

#include "../utils/utils.h"
#include "../layer/layer.h"
#include "../utils/global.h"

using namespace layer;

namespace optimizer
{
	class Optimizer
	{
	public:
		Optimizer(float _learn_rate, float _weight_decay);
		virtual ~Optimizer();
		void optimize() {};

		float learn_rate;
		float weight_decay;
		float inv_weight_decay;
	};
}



#endif // !_OPTIMIZER_H_

