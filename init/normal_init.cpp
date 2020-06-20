#include "normal_init.h"

namespace init
{

	NormalInit::NormalInit(float _mean, float _std) : Init()
	{
		std = _std;
		mean = _mean;
	}

	void NormalInit::initilize(Layer* _layer)
	{
		if (_layer->param_size > 0)
		{
			net_utils::setGpuNormalValue(_layer->param, _layer->param_size, mean, std);
		}

		if (_layer->param_bias_size > 0)
		{
			net_utils::setGpuValue(_layer->param_bias, _layer->param_bias_size, 0);
		}
	}

	NormalInit::~NormalInit()
	{
	}


}