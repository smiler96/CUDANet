#include "xavier_init.h"

namespace init
{
	
	XavierInit::XavierInit(float _gain) : Init()
	{
		gain = _gain;
	}

	void XavierInit::initilize(Layer* _layer)
	{
		if (_layer->param_size > 0)
		{
			get_fan_in_and_fan_out(_layer);

			std = gain * std::sqrt(2.0 / ((float)fan_in + (float)fan_out));

			net_utils::setGpuNormalValue(_layer->param, _layer->param_size, 0, std);
		}


		if (_layer->param_bias_size > 0)
		{
			net_utils::setGpuValue(_layer->param_bias, _layer->param_bias_size, 0);
		}
	}

	XavierInit::~XavierInit()
	{
	}


}