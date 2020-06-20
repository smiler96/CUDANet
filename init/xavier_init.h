#pragma once
#ifndef _XAVIER_INIT_H_
#define _XAVIER_INIT_H_

#include "init.h"

namespace init
{
	class XavierInit : public Init
	{
	public:
		XavierInit(float _gain);
		~XavierInit();

		void initilize(Layer* _layer);

		float gain;

	private:

	};

}

#endif // !_XAVIER_INIT_H_

