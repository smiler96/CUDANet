#pragma once
#ifndef _KAIMING_H_
#define _KAIMING_H_

#include "init.h"

#define KM_FANIN_MODE 0
#define KM_FANOUT_MODE 1

namespace init
{
	class KaiMingInit : public Init
	{
	public:
		KaiMingInit(int _mode);
		~KaiMingInit();

		void initilize(Layer* _layer);

		int mode;

	private:

	};

}

#endif // !_KAIMING_H_
