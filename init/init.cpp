#include "init.h"

namespace init
{

	Init::Init()
	{
	}

	void Init::get_fan_in_and_fan_out(Layer* _layer)
	{
		std::string layerName = _layer->getLayerName();
		
		int _c_in, _c_out, _kh, _kw;
		cudnnDataType_t _temp_t;
		cudnnTensorFormat_t _temp_f;

		if (layerName == "conv")
		{
			callCuda(cudnnGetFilter4dDescriptor(((Convolution*)_layer)->filter_desc,
				&_temp_t,
				&_temp_f,
				&_c_out, &_c_in, &_kh, &_kw));
		}
		else if (layerName == "deconv")
		{
			callCuda(cudnnGetFilter4dDescriptor(((Deconvolution*)_layer)->filter_desc,
				&_temp_t,
				&_temp_f,
				&_c_in, &_c_out, &_kh, &_kw));
		}
		else if (layerName == "batchnorm")
		{
			_c_in = _layer->param_size;
			_c_out = _layer->param_size;
			_kh = 1;
			_kw = 1;
		}

		fan_in = _c_in * _kh * _kw;
		fan_out = _c_out * _kh * _kw;
	}

	Init::~Init()
	{
	}
}


