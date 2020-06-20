/*
class BatchNorm inherits from Layer
for conv spatial batch normalization
20191219
created by wanqian
*/

#ifndef _BATCH_NORMALIZATION_H_
#define _BATCH_NORMALIZATION_H_

#include "layer.h"
#include "../utils/global.h"

using namespace global;

namespace layer
{

	class BatchNorm : public Layer
	{
	public:
		BatchNorm(Layer* _prev, int _channels, float _epsilon, float _expAverFactor, float alpha, float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);
		virtual ~BatchNorm();

		void forward(bool train = true);
		void backward();
		void update();

		int channels;
		float*  bnScale;
		float*  bnBias;
		float*  bnScaleDiff;
		float*  bnBiasDiff;

		float*  resultMovingMean;
		float*  resultMovingVar;
		float*  resultSaveMean;
		float*  resultSaveInvVar;

	public:
		float epsilon;
		float expAverFactor;

		cudnnBatchNormMode_t bnMode;
		cudnnTensorDescriptor_t bnScaleBiasMeanVarDiscriptor;

		cudnnTensorDescriptor_t resultMovingMeanDiscriptor;
		cudnnTensorDescriptor_t resultMovingVarDiscriptor;

		cudnnTensorDescriptor_t resultSaveMeanDiscriptor;
		cudnnTensorDescriptor_t resultSaveVarDiscriptor;
	};
}


#endif // !_BATCH_NORMALIZATION_H_
