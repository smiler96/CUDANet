#include "kaiming_init.h"

namespace init
{

	KaiMingInit::KaiMingInit(int _mode) : Init()
	{
		mode = _mode;
	}

	void KaiMingInit::initilize(Layer* _layer)
	{
		if (_layer->param_size > 0)
		{
			get_fan_in_and_fan_out(_layer);

			if(mode == KM_FANIN_MODE)
				std = std::sqrt(2.0 / ((float)fan_in));
			else
				std = std::sqrt(2.0 / ((float)fan_out));

			net_utils::setGpuNormalValue(_layer->param, _layer->param_size, 0, std);
		}


		if (_layer->param_bias_size > 0)
		{
			net_utils::setGpuValue(_layer->param_bias, _layer->param_bias_size, 0);
		}
	}

	KaiMingInit::~KaiMingInit()
	{
	}


}