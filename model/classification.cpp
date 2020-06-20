#include "classification.h"
#include "../optimizer/adam_optimizer.h"

namespace model{
	Classification::Classification(float* _data, int _data_dim, float* _label, int _label_dim,
		int _train_size, int _val_size, int _batch) : 
		Network(_data, _data_dim, _label, _train_size, _val_size, _batch){

		//size = _train_size;
		//val_size = _val_size;
		label_dim = _label_dim;
		train_error = 100;
		val_error = 100;

		callCuda(cudaMalloc(&label, sizeof(float) * label_dim * batch));
		callCuda(cudaMemcpy(label, h_label, sizeof(float) * batch, cudaMemcpyHostToDevice));
	}

	Classification::~Classification() {
		callCuda(cudaFree(label));
	}

	void Classification::PushOutput(int label_dim) {
		Output* output = new Output(layers.back(), label, label_dim, batch);
		layers.push_back(output);
	}

	void Classification::Train(int iteration, float half_time, float half_rate,
		float step_decrease, bool debug) {

		int GLOBAL_ADAMP_ITERA = 1;
		optimizer::Adam adam(-1e-3, 0.0005f, 0.9, 0.999, 1e-8);
		// train the network multiple times
		
		std::vector<int> vDataIndex;
		for (int i = 0; i<size; ++i) vDataIndex.push_back(i); 

		for (int k = 0; k < iteration && lambda > 5e-3; k++) {
			if (debug)
				for (int i = layers.size() - 1; i > 0; i--) {
					if (layers[i]->param_size != 0)
						net_utils::printGpuMax(layers[i]->param, layers[i]->param_size);
				}

			// divide the training set to small pieces
			float* predict = new float[size];
			int offset = 0;
			std::cout << "Iteration " << k + 1 << std::endl;
			std::random_shuffle(vDataIndex.begin(), vDataIndex.end());
			for (int b = 0; b < size / batch; b++) {
				/*for (int id = 0; id < batch; ++id)
				{
					callCuda(cudaMemcpy(data + data_dim * id, h_data + (vDataIndex.at(offset + id)) * data_dim, sizeof(float) * data_dim, cudaMemcpyHostToDevice));
					callCuda(cudaMemcpy(label + id, h_label + (vDataIndex.at(offset + id)), sizeof(float) * 1, cudaMemcpyHostToDevice));
				}*/

				// choose a new piece and its labels
				callCuda(cudaMemcpy(data, h_data + offset * data_dim, sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
				callCuda(cudaMemcpy(label, h_label + offset, sizeof(float) * batch, cudaMemcpyHostToDevice));

				// forward propagation
				for (int i = 0; i < layers.size(); i++)
					layers[i]->forward(true);

				callCuda(cudaMemcpy(predict + offset, ((SoftmaxAnL*)layers[layers.size() - 1])->predict_label, sizeof(float) * batch, cudaMemcpyDeviceToHost));
				// calculate the predict error of every sample in training set
				// back propagation
				for (int i = layers.size() - 1; i > 0; i--) {
					layers[i]->backward();
					//layers[i]->update(); // update the parameters
					adam.optimize(layers[i], GLOBAL_ADAMP_ITERA);
				}
				offset += batch;

			}
			std::cout << "label and predict:\n";
			net_utils::printGpuMatrix(label, 5, 1, 5, 2);
			net_utils::printCpuMatrix(predict, 5, 1, 5, 2);
			std::cout << "\n";
			int errors = 0;
			for (int i = 0; i < size; i++)
				if (abs(h_label[i] - predict[i]) > 0.1)
					errors++;
			train_error = errors * 100.0 / size;
			std::cout << "Train error: " << train_error << std::endl;
			delete[] predict;

			for (int i = layers.size() - 1; i > 0; i--)
				layers[i]->adjust_learning(step_decrease);

			// validation error
			if (val_size > 0) {
				float* predict = new float[val_size];
				offset = 0;
				for (int b = 0; b < val_size / batch; b++) {
					callCuda(cudaMemcpy(data, h_data + (size + offset) * data_dim,
						sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
					for (int i = 0; i < layers.size(); i++)
						layers[i]->forward(false);
					callCuda(cudaMemcpy(predict + offset * label_dim,
						layers[layers.size() - 1]->data,
						sizeof(float) * label_dim * batch, cudaMemcpyDeviceToHost));
					offset += batch;
				}
				int errors = 0;
				for (int i = 0; i < val_size; i++)
					if (abs(h_label[size + i] - predict[i]) > 0.1)
						errors++;

				float prev_error = val_error;
				val_error = errors * 100.0 / val_size;
				std::cout << "Validation error: " << val_error << std::endl;

				// adjust the learning rate if the validation error stabilizes

				if ((prev_error - val_error) / prev_error < half_time) {
					lambda *= half_rate;
					std::cout << "-- Learning rate decreased --" << std::endl;
					for (int i = layers.size() - 1; i > 0; i--)
						layers[i]->adjust_learning(half_rate);
				}

				delete[] predict;
			}

			GLOBAL_ADAMP_ITERA++;
		}

	}

	void Classification::Test(float* predict_label) {
		int offset = 0;
		for (int b = 0; b < size / batch; b++) {
			callCuda(cudaMemcpy(data, h_data + offset * data_dim,
				sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
			for (int i = 0; i < layers.size(); i++)
				layers[i]->forward(false);
			callCuda(cudaMemcpy(predict_label + offset * label_dim,
				layers[layers.size() - 1]->data,
				sizeof(float) * label_dim * batch, cudaMemcpyDeviceToHost));
			offset += batch;
		}
	}
}













//#include "classification.h"
//#include "../optimizer/adam_optimizer.h"
//
//namespace model {
//	Classification::Classification(float* _data, int _data_dim, float* _label, int _label_dim,
//		int _train_size, int _val_size, int _batch) :
//		Network(_data, _data_dim, _label, _train_size, _val_size, _batch) {
//
//		//size = _train_size;
//		//val_size = _val_size;
//		label_dim = _label_dim;
//		train_error = 100;
//		val_error = 100;
//
//		callCuda(cudaMalloc(&label, sizeof(float) * label_dim * batch));
//		callCuda(cudaMemcpy(label, h_label, sizeof(float) * label_dim * batch,
//			cudaMemcpyHostToDevice));
//	}
//
//	Classification::~Classification() {
//		callCuda(cudaFree(label));
//	}
//
//	void Classification::PushOutput(int label_dim) {
//		Output* output = new Output(layers.back(), label, label_dim, batch);
//		layers.push_back(output);
//	}
//
//	void Classification::Train(int iteration, float half_time, float half_rate,
//		float step_decrease, bool debug) {
//
//		// train the network multiple times
//		float Loss = 0.0f;
//		float temp_loss = 0.0f;
//		int GLOBAL_ADAMP_ITERA = 1;
//		optimizer::Adam adam(-1e-3, 0.0005f, 0.9, 0.999, 1e-8);
//
//		for (int k = 0; k < iteration && lambda > 5e-3; k++) {
//			//if (debug)
//			//	for (int i = layers.size() - 1; i > 0; i--) {
//			//		if (layers[i]->param_size != 0)
//			//			net_utils::printGpuMax(layers[i]->param, layers[i]->param_size);
//			//	}
//
//			Loss = 0;
//			int offset = 0;
//			std::cout << "Iteration " << k + 1 << std::endl;
//
//			for (int b = 0; b < size / batch; b++) {
//				//if (debug)
//					//for (int i = layers.size() - 1; i > 0; i--) {
//					//	if (layers[i]->param_size != 0)
//					//		net_utils::printGpuMax(layers[i]->param, layers[i]->param_size);
//					//}
//
//					// choose a new piece and its labels
//				callCuda(cudaMemcpy(data, h_data + offset * data_dim,
//						sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
//				callCuda(cudaMemcpy(label, h_label + offset * label_dim,
//					sizeof(float) * label_dim * batch, cudaMemcpyHostToDevice));
//
//				// forward propagation
//				for (int i = 0; i < layers.size(); i++)
//					layers[i]->forward(true);
//
//				if (!b)
//				{
//					std::cout << "softmax input z:" << std::endl;
//					net_utils::printGpuMatrix(layers[layers.size() - 1]->prev->data, 100, 10, 10);
//					std::cout << "softmax output a:" << std::endl;
//					net_utils::printGpuMatrix(((SoftmaxAnL*)layers[layers.size() - 1])->tmp_data, 100, 10, 10);
//					std::cout << "loss:" << std::endl;
//					net_utils::printGpuMatrix(layers[layers.size() - 1]->data, 1, 1);
//				}
//
//
//
//				// ¼ÆËãËðÊ§ cla_loss
//				float temp_loss = 0.0f;
//				callCuda(cudaMemcpy(&temp_loss, layers[layers.size() - 1]->data, sizeof(float) * 1, cudaMemcpyDeviceToHost));
//				Loss += temp_loss;
//
//				// back propagation
//				for (int i = layers.size() - 1; i > 0; i--) {
//					layers[i]->backward();
//					//layers[i]->update(); // update the parameters
//					adam.optimize(layers[i], GLOBAL_ADAMP_ITERA);
//
//				}
//				offset += batch;
//			}
//			Loss /= size;
//			std::cout << "Loss =  " << Loss << std::endl;
//
//			GLOBAL_ADAMP_ITERA++;
//		}
//
//	}
//
//	void Classification::Test(float* predict_label) {
//
//	}
//}

