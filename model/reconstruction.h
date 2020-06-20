/*
class classification inherits from Network
for classification model
20191103
created by wanqian 
*/

#ifndef _H_RECONSTRUCTION_H_
#define _H_RECONSTRUCTION_H_

#include "network.h"
#include "../layer/cluster.h"
#include "../utils/image.h"
#include "../optimizer/adam_optimizer.h"
#include "../optimizer/sgd_optimizer.h"

namespace model {

	class Reconstruction : public Network
	{
	public:
		// _label = _data, label is the same as data for reconstruction network
		Reconstruction(float* _data, int _data_dim, float* _label,
			int _train_size, int _val_size, int _batch);

		Reconstruction(int _batch);
		Reconstruction(int _train_size, int _val_size, int _batch);

		//std::vector<Network*> LayerBranches; // layer层的分支
		std::map<std::string, std::vector<Reconstruction*>> Branches; // 所有layer层的分支

		// Network(Layer* layer);
		void PushBranchNet(Layer* layer, float* _label_dev);

		virtual ~Reconstruction();

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
		void Train(string paras_path, int iteration, int step_iter=100, float step_decrease = 0.1, bool debug=false);

		/*
		* Test the network, used after switching the test data
		*
		* predict_label: label array to store the reconstructed data in cpu, not gpu
		*/
		void Test(std::vector<float*> reconstruction_data);

		void PushCluster(int K, float cluster_weight, float alpha, float* init_param, float sigma = 0.01f,
			float momentum = 0.9f, float weight_decay = 0);

		void ReadParams(std::string dir);

		void SaveParams(std::string dir);

		float* reconstruction_data;

	private:
		float loss; // previous error rate
	};

}


#endif // !_H_CLASSIFICATION_H_
