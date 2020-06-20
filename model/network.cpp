#include "network.h"

using namespace layer;
using namespace std;

namespace model {

	Network::Network(float* _data, int _data_dim, float* _label, 
		int _train_size, int _val_size, int _batch) {
		h_data = _data; // data in cpu
		h_label = _label;
		size = _train_size;
		val_size = _val_size;
		batch = _batch;
		data_dim = _data_dim;
		//label_dim = _label_dim;
		lambda = 1;
		loss_weight = 1.0f;

		callCuda(cudaMalloc(&data, sizeof(float) * data_dim * batch));
		callCuda(cudaMemcpy(data, h_data, sizeof(float) * data_dim * batch,
							cudaMemcpyHostToDevice));
		/*callCuda(cudaMalloc(&label, sizeof(float) * label_dim * batch));
		callCuda(cudaMemcpy(label, h_label, sizeof(float) * label_dim * batch,
							cudaMemcpyHostToDevice));*/
	}
	
	Network::Network(int _batch)
	{
		batch = _batch;
	}

	Network::Network(int _train_size, int _val_size, int _batch)
	{
		size = _train_size;
		val_size = _val_size;
		batch = _batch;  
		lambda = 1;
		
		convNum = 0; 
		poolNum = 0;  
		deconvNum = 0;  
		activateNum = 0;  
		bnNum = 0;  
	}

	void Network::PushBranchNet(Layer* layer, float* _label_dev)
	{
		int _n, _c, _h, _w, _tmp;
		cudnnDataType_t _t;
		callCudnn(cudnnGetTensor4dDescriptor(layer->t_data, &_t, &_n, &_c, &_h, &_w, &_tmp,
				&_tmp, &_tmp, &_tmp));
				
		Network* branch = new Network(_n);
		
		branch->data = layer->data;
		branch->label = _label_dev;
		branch->PushInput(_c, _h, _w);
		
		if (layer->isBranch() == false)
		{
			layer->setBranch();
			std::vector<Network*> LayerBranches; // layer²ãµÄ·ÖÖ§
			LayerBranches.push_back(branch);
			Branches.insert(std::pair<string, std::vector<Network*>>(layer->getLayerName() + "_" + std::to_string(layer->getLayerId()), LayerBranches));
		}
		else
		{
			Branches[layer->getLayerName() + "_" + std::to_string(layer->getLayerId())].push_back(branch);
		}
			
	}
	
	void Network::InputTrainData(float* _data, int _data_dim, float* _label, int _label_dim)
	{
		data_dim = _data_dim;
		h_data = _data; // data in cpu
		h_label = _label;
		
		callCuda(cudaMalloc(&data, sizeof(float) * data_dim * batch));
		//callCuda(cudaMemcpy(data, h_data, sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
		callCuda(cudaMalloc(&label, sizeof(float) * _label_dim * batch));
		//callCuda(cudaMemcpy(label, h_label, sizeof(float) * _label_dim * batch, cudaMemcpyHostToDevice));
	}
	
	Network::~Network() {
		h_data = NULL;
		h_label = NULL;
		callCuda(cudaFree(data));
		callCuda(cudaFree(label));
		for (std::map<string, std::vector<Network*>>::iterator bh = Branches.begin(); bh != Branches.end(); ++bh)
			for (Network* layer_bh : bh->second)
				for (Layer* l : layer_bh->layers)
					delete l;

		for (Layer* l : layers)
			delete l;
	}

	pair<float*, float*> Network::GetData() {
		return make_pair(data, label);
	}

	/*void Network::Train(int iteration, float half_time, float half_rate,
						float step_decrease, bool debug) {

		// train the network multiple times
		for (int k = 0; k < iteration && lambda > 5e-3; k++) {
			if (debug)
				for (int i = layers.size() - 1; i > 0; i--) {
					if (layers[i]->param_size != 0)
						net_utils::printGpuMax(layers[i]->param, layers[i]->param_size);
				}

			// divide the training set to small pieces
			int offset = 0;
			std::cout << "Iteration " << k + 1 << std::endl;
			for (int b = 0; b < size / batch; b++) {

				// choose a new piece and its labels
				callCuda(cudaMemcpy(data, h_data + offset * data_dim,
									sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
				callCuda(cudaMemcpy(label, h_label + offset * label_dim,
									sizeof(float) * label_dim * batch, cudaMemcpyHostToDevice));

				// forward propagation
				for (int i = 0; i < layers.size() - 1; i++)
					layers[i]->forward();

				// back propagation
				for (int i = layers.size() - 1; i > 0; i--) {
					layers[i]->backward();
					layers[i]->update(); // update the parameters
				}
				offset += batch;

			}

			for (int i = layers.size() - 1; i > 0; i--)
				layers[i]->adjust_learning(step_decrease);

			// training error
			if (size > 0) {
				float* predict = new float[size];
				offset = 0;
				for (int b = 0; b < size / batch; b++) {
					callCuda(cudaMemcpy(data, h_data + offset * data_dim,
										sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
					for (int i = 0; i < layers.size(); i++)
						layers[i]->forward(false);
					callCuda(cudaMemcpy(predict + offset * label_dim,
										layers[layers.size() - 1]->data,
										sizeof(float) * label_dim * batch, cudaMemcpyDeviceToHost));
					offset += batch;
				}
				// calculate the predict error of every sample in training set
				int errors = 0;
				for (int i = 0; i < size; i++)
					if (abs(h_label[i] - predict[i]) > 0.1)
						errors++;

				train_error = errors * 100.0 / size;
				std::cout << "Train error: " << train_error << std::endl;
				delete[] predict;
			}

			// validation error

			if (val_size > 0) {
				float* predict = new float[val_size];
				offset = 0;
				for (int b = 0; b < val_size / batch; b++) {
					callCuda(cudaMemcpy(data, h_data + (size  + offset) * data_dim,
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
		}

	}*/

	void Network::SetLossWeight(const float _loss_weight)
	{
		loss_weight = _loss_weight;
	}

	void Network::PushInput(int c, int h, int w) {
		Input* input = new Input(batch, c, h, w, data);
		input->setLayerName("input");
		input->setLayerId(0);

		layers.push_back(input);
	}

	/*void Network::PushOutput(int label_dim) {
		Output* output = new Output(layers.back(), label, label_dim, batch);
		layers.push_back(output);
	}*/

	void Network::PushConvolution(int c, int kernel, int stride, int padding, float alpha, float sigma,
								  float momentum, float weight_decay) 
	{
		convNum++;
		
		Convolution* conv = new Convolution(layers.back(), batch, c, kernel, stride, padding, alpha,
											sigma, momentum, weight_decay);
		conv->setLayerName("conv");
		conv->setLayerId(convNum);
		layers.push_back(conv);
	}

	void Network::PushDeconvolution(int c, int kernel, int stride, int padding, float alpha, float sigma,
		float momentum, float weight_decay) 
	{
		deconvNum++;
		
		Deconvolution* deconv = new Deconvolution(layers.back(), batch, c, kernel, stride, padding, alpha,
			sigma, momentum, weight_decay);
			
		deconv->setLayerName("deconv");
		deconv->setLayerId(deconvNum);
		layers.push_back(deconv);
	}

	void Network::PushPooling(int size, int stride)
	{
		poolNum++;
		
		Pooling* pool = new Pooling(layers.back(), size, stride);
		pool->setLayerName("pooling");
		pool->setLayerId(poolNum);
		layers.push_back(pool);
	}

	void Network::PushFullyConnected(int output_size, float dropout_rate, float alpha,
		float sigma, float momentum, float weight_decay)
	{
		fullyconnectedNum++;
		FullyConnected* FC = new FullyConnected(layers.back(), output_size, dropout_rate, 
			alpha, sigma, momentum, weight_decay);
		FC->setLayerName("fullyconnected");
		FC->setLayerId(fullyconnectedNum);
		layers.push_back(FC);
	}

	void Network::PushActivation(cudnnActivationMode_t mode) 
	{
		activateNum++;
		
		Activation* activation = new Activation(layers.back(), mode);
		activation->setLayerName("activation");
		activation->setLayerId(activateNum);
		layers.push_back(activation);
	}

	void Network::PushReLU(int output_size, float dropout_rate, float alpha,
						   float sigma, float momentum, float weight_decay) 
	{
		activateNum++;
		ReLU* relu = new ReLU(layers.back(), output_size, dropout_rate, alpha,
							  sigma, momentum, weight_decay);
		relu->setLayerName("activation");
		relu->setLayerId(activateNum);
		layers.push_back(relu);
	}

	/*void Network::PushSoftmax(int output_size, float dropout_rate, float alpha,
							  float sigma, float momentum, float weight_decay) {
		Softmax* softmax = new Softmax(layers.back(), output_size, dropout_rate, alpha,
									   sigma, momentum, weight_decay);
		softmax->setLayerName("softmax");
		layers.push_back(softmax);
	}*/

	void Network::PushSoftmaxAnL(int _class_num)
	{
		SoftmaxAnL* softmaxanl = new SoftmaxAnL(layers.back(), label, _class_num, batch);
		softmaxanl->setLayerName("softmaxanl");
		layers.push_back(softmaxanl);
	}

	void Network::PushEuclideanLoss() {
		EuclideanLoss* loss = new EuclideanLoss(layers.back(), label, loss_weight);
		layers.push_back(loss);
	}

	//void Network::PushSoftmaxLoss(int _class_num, int _batch, float* _label) {
	/*void Network::PushSoftmaxLoss(int _class_num) {
		SoftmaxLoss* loss = new SoftmaxLoss(layers.back(), this->label, _class_num, batch);
		layers.push_back(loss);
	}*/

	void Network::PushBatchNorm(int _channels, float _epsilon, float _expAverFactor, float alpha, float sigma, float momentum, float weight_decay)
	{
		bnNum++;
		BatchNorm* bn = new BatchNorm(layers.back(), _channels, _epsilon, _expAverFactor, alpha, 
			sigma, momentum, weight_decay);
		bn->setLayerName("batchnorm");
		bn->setLayerId(bnNum);
		layers.push_back(bn);
	}

	void Network::Pop() {
		Layer* tmp = layers.back();
		layers.pop_back();
		delete tmp;
		layers.back()->next = NULL;
	}

	void Network::SwitchData(float* h_data, float* h_label, int count) {
		// switch data without modifying the batch size
		size = count;
		this->h_data = h_data;
		this->h_label = h_label;
	}

	void Network::SwitchDataD(float* data, float* label, int count) {
		// switch data without modifying the batch size
		size = count;
		this->data = data;
		this->label = label;
	}

	/*void Network::Test(float* label) {
		int offset = 0;
		for (int b = 0; b < size / batch; b++) {
			callCuda(cudaMemcpy(data, h_data + offset * data_dim,
								sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
			for (int i = 0; i < layers.size(); i++)
				layers[i]->forward(false);
			callCuda(cudaMemcpy(label + offset * label_dim,
								layers[layers.size() - 1]->data,
								sizeof(float) * label_dim * batch, cudaMemcpyDeviceToHost));
			offset += batch;
		}
	}*/

	void Network::PrintGeneral() {
		std::cout << "Neural Network" << std::endl;
		std::cout << "Layers: " << layers.size() << std::endl;
		int i = 0;
		for (Layer* l : layers)
			std::cout << " - " << i++ << ' ' << l->data_size << ' ' << l->param_size << std::endl;
	}

	void Network::PrintData(int offset, int r, int c, int precision) {
		net_utils::printGpuMatrix(data + offset, r * c, r, c, precision);
	}

	void Network::ReadParams(std::string dir) {
		for (int i = 1; i < layers.size() - 1; i++) {
			if (layers[i]->param_size > 0)
				net_utils::readGPUMatrix(dir + std::to_string(i), layers[i]->param, layers[i]->param_size);
			if (layers[i]->param_bias_size > 0)
				net_utils::readGPUMatrix(dir + std::to_string(i) + "_bias",
									 layers[i]->param_bias, layers[i]->param_bias_size);
		}
	}

	void Network::SaveParams(std::string dir) {
		for (int i = 1; i < layers.size() - 1; i++) {
			if (layers[i]->param_size > 0)
				net_utils::writeGPUMatrix(dir + std::to_string(i), layers[i]->param,
									  layers[i]->param_size);
			if (layers[i]->param_bias_size > 0)
				net_utils::writeGPUMatrix(dir + std::to_string(i) + "_bias",
									  layers[i]->param_bias, layers[i]->param_bias_size);
		}
		std::cout << "Params saved." << std::endl;
	}

	void Network::InitParas(std::string InitType) {
		init::Init *initor;
		if (InitType == "xavier")
			initor = new init::XavierInit(1.0);
		else if (InitType == "kaiming")
			initor = new init::KaiMingInit(0);
		else
			std::cout << "choose xavier or kaiming init mode.\n";

		for (int i = 1; i < layers.size() - 1; i++) {
			if (layers[i]->param_size > 0)
			{
				initor->initilize(layers[i]);
			}
		}

		delete initor;
		std::cout << "Init Params done." << std::endl;
	}

} /* namespace model */
