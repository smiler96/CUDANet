#pragma once
#ifndef _NORMAL_INIT_H_
#define _NORMAL_INIT_H_

#include "init.h"

namespace init
{
	class NormalInit : public Init
	{
	public:
		NormalInit(float _mean, float _std);
		~NormalInit();

		void initilize(Layer* _layer);

		int mode;

	private:

	};

}

#endif // !_NORMAL_INIT_H_

