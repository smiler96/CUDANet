/* 
class classification inherits from Network
for classification model
20191102
created by wanqian 
*/

#ifndef _H_CLASSIFICATION_H_
#define _H_CLASSIFICATION_H_

#include "network.h"
namespace model {

	class Classification : public Network
	{
	public:
		Classification(float* _data, int _data_dim, float* _label, int _label_dim,
			int _train_size, int _val_size, int _batch);

		virtual ~Classification();

		/*
		* Insert Output layer (should be called at the end for classification)
		*
		* label_count:  label dimension (example: 10 for mnist digits)
		*/
		void PushOutput(int label_count);

		/*
		* Train the network: the start may be slow, need to change sigma of initial
		* weight or adjust learning rate, etc.
		*
		* iteration: number of epoch
		* half_time: threshold for changing the learning rate
		* half_rate: learning rate adjustment
		* step_decrease: decrease learning rate by each batch
		* debug: debug mode (print some extra information)
		*/
		void Train(int iteration, float half_time = 0, float half_rate = 0.5,
			float step_decrease = 1, bool debug = false);

		/*
		* Test the network, used after switching the test data
		*
		* predict_label: label array to store the prediction in cpu, not gpu
		*/
		void Test(float* predict_label);

	private:
		float train_error; // previous error rate
		float val_error;
		int label_dim; // dimension of one label (usually 1 (0, 1, 2, 3, 4, 5, 6, 7, ...))
		//int size, val_size;
	};

	}


#endif // !_H_CLASSIFICATION_H_
