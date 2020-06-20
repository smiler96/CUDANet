#pragma once
#ifndef _INIT_H_
#define _INIT_H_

#include "../utils/set_value.h"
#include "../layer/layer.h"
#include "../layer/convolution.h"
#include "../layer/deconvolution.h"
#include "../layer/batch_normalization.h"

using namespace layer;

namespace init
{
	class Init
	{
	public:
		Init();
		virtual ~Init();

		virtual void initilize(Layer* _layer) = 0;

		int fan_in;
		int fan_out;

		float std;
		float mean;

		void get_fan_in_and_fan_out(Layer* _layer);

	private:

	};

}


#endif // !_INIT_H_

