#include "reconstruction.h"

namespace model {

	Reconstruction::Reconstruction(float* _data, int _data_dim, float* _label,
		int _train_size, int _val_size, int _batch) : 
		Network(_data, _data_dim, _label, _train_size, _val_size, _batch) {
		
		loss = 100.0f;
		reconstruction_data = new float[data_dim * size];

		callCuda(cudaMalloc(&label, sizeof(float) * _data_dim * batch));
		callCuda(cudaMemcpy(label, h_label, sizeof(float) * _data_dim * batch,
			cudaMemcpyHostToDevice));
	}

	Reconstruction::Reconstruction(int _batch) : Network(_batch)
	{
		batch = _batch;
	}

	Reconstruction::Reconstruction(int _train_size, int _val_size, int _batch) : 
		Network(_train_size, _val_size, _batch)
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

	void Reconstruction::PushBranchNet(Layer* layer, float* _label_dev)
	{
		int _n, _c, _h, _w, _tmp;
		cudnnDataType_t _t;
		callCudnn(cudnnGetTensor4dDescriptor(layer->t_data, &_t, &_n, &_c, &_h, &_w, &_tmp,
			&_tmp, &_tmp, &_tmp));

		Reconstruction* branch = new Reconstruction(_n);

		branch->data = layer->data;
		branch->label = _label_dev;

		branch->PushInput(_c, _h, _w);

		if (layer->isBranch() == false)
		{
			layer->setBranch();
			std::vector<Reconstruction*> LayerBranches; // layer层的分支
			LayerBranches.push_back(branch);
			Branches.insert(std::pair<string, std::vector<Reconstruction*>>(layer->getLayerName() + "_" + std::to_string(layer->getLayerId()), LayerBranches));
		}
		else
		{
			Branches[layer->getLayerName() + "_" + std::to_string(layer->getLayerId())].push_back(branch);
		}

	}

	Reconstruction::~Reconstruction() {
		delete[] reconstruction_data;
		callCuda(cudaFree(label));
		for (std::map<string, std::vector<Reconstruction*>>::iterator bh = Branches.begin(); bh != Branches.end(); ++bh)
			for (Reconstruction* layer_bh : bh->second)
				for (Layer* l : layer_bh->layers)
					delete l;

		for (Layer* l : layers)
			delete l;
	}

	void Reconstruction::Train(string paras_path, int iteration, int step_iter, float step_decrease, bool debug)
	{
		// train the network multiple times
		reconstruction_data = new float[data_dim * size];
		float prev_loss = 100.0f;
		std::vector<float> loss;

		int batchNum = size / batch;
		std::vector<int> vDataIndex;
		for (int i = 0; i<size; ++i) vDataIndex.push_back(i); // offset += batch

		//using namespace optimizer;
		int GLOBAL_ADAMP_ITERA = 1;
		optimizer::Adam adam(-1e-3, 0.0005f, 0.9, 0.999, 1e-8);
		optimizer::SGD sgd(-1, 0.0005f);
		int k;
		for (k = 1; k < iteration; k++) {
			if (debug)
				for (int i = layers.size() - 1; i > 0; i--)
				{
					if (layers[i]->param_size != 0)
						net_utils::printGpuMax(layers[i]->param, layers[i]->param_size);
				}

			loss.clear();
			loss.push_back(0);
			for (std::map<string, std::vector<Reconstruction*>>::iterator bh = Branches.begin(); bh != Branches.end(); ++bh)
				for (Reconstruction* layer_bh : bh->second)
				{
					loss.push_back(0);
				}

			int offset = 0;
			std::cout << "Iteration " << k << std::endl;
			//for (int b = 0; b < size / batch; b++) 
			std::random_shuffle(vDataIndex.begin(), vDataIndex.end());
			for (int b = 0; b < batchNum; ++b)
			{
				// choose a new piece and its labels
				for (int id = 0; id < batch; ++id)
				{
					callCuda(cudaMemcpy(data + data_dim * id, h_data + (vDataIndex.at(offset + id)) * data_dim, sizeof(float) * data_dim, cudaMemcpyHostToDevice));
					callCuda(cudaMemcpy(label + data_dim * id, h_label + (vDataIndex.at(offset + id)) * data_dim, sizeof(float) * data_dim, cudaMemcpyHostToDevice));
				}
				/*callCuda(cudaMemcpy(data, h_data + offset * data_dim, sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
				callCuda(cudaMemcpy(label, h_label + offset * data_dim, sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));*/

				// forward propagation
				int lossN = 1;
				for (int n = 1; n < layers.size(); n++)
				{
					layers[n]->forward(true);

					std::string lName = (layers[n])->getLayerName() + "_" + std::to_string((layers[n])->getLayerId());
					if (layers[n]->isBranch())
					{
						std::vector<Reconstruction*>* branch = &(Branches[lName]);
						for (int br_i = 0; br_i < branch->size(); br_i++)
						{
							Reconstruction* recNet = branch->at(br_i);
							for (int l_j = 1; l_j < recNet->layers.size(); l_j++)
							{
								(recNet->layers[l_j])->forward(true);
							}
							loss[lossN] += ( *(((recNet->layers).back())->data) ) / batch / (float)data_dim;
							lossN++;
						}			
					}

					//layers[n]->forward(true);
				}

				//最后一层时损失层，一般为 Euclidean loss
				//callCuda(cudaMemcpy(batch_loss, layers[layers.size() - 1]->data, 1, cudaMemcpyDeviceToHost));
				loss[0] += (*(layers[layers.size() - 1]->data)) / batch / (float)data_dim;
				if (offset == 0 && (k-1) % 50 == 0)
				{
					SaveParams(paras_path);

					// 倒数第二层才是重构层，最后一层时损失层，一般为 Euclidean loss
					callCuda(cudaMemcpy(reconstruction_data + offset * data_dim, layers[layers.size() - 2]->data,
						sizeof(float) * data_dim * batch, cudaMemcpyDeviceToHost));
					std::string str = "C:\\Users\\ms952\\Desktop\\lcd\\branch_" + std::to_string(k) + "_0.bmp";
					net_utils::saveImage(str, reconstruction_data, 256, 256, 1, 0);

					callCuda(cudaMemcpy(reconstruction_data + offset * data_dim, ((EuclideanLoss*)layers[layers.size() - 1])->label,
						sizeof(float) * data_dim * batch, cudaMemcpyDeviceToHost));
					net_utils::saveImage("C:\\Users\\ms952\\Desktop\\lcd\\branch_" + std::to_string(k) + 
						"_0_label.bmp" , reconstruction_data, 256, 256, 1, 0);

					int cn = 1;
					for (std::map<string, std::vector<Reconstruction*>>::iterator bh = Branches.begin(); bh != Branches.end(); ++bh)
						for (Reconstruction* layer_bh : bh->second)
						{
							callCuda(cudaMemcpy(reconstruction_data + offset * data_dim, layer_bh->layers[layer_bh->layers.size() - 2]->data,
								sizeof(float) * data_dim * batch, cudaMemcpyDeviceToHost));
							str = "C:\\Users\\ms952\\Desktop\\lcd\\branch_" + std::to_string(k) + "_" + std::to_string(cn) + ".bmp";
							net_utils::saveImage(str, reconstruction_data, 256, 256, 1, 0);

							//callCuda(cudaMemcpy(reconstruction_data + offset * data_dim, ((EuclideanLoss*)layer_bh->layers[layer_bh->layers.size() - 1])->label,
								//sizeof(float) * data_dim * batch, cudaMemcpyDeviceToHost));
							//net_utils::saveImage("D:/GitHub/VGG_XNet_CUDNN_CUDA/VGG_XNet_CUDNN_CUDA/data/rec_KTD_scarf/branch_" + std::to_string(k) + 
								//"_" + std::to_string(cn) + "_label.bmp", reconstruction_data, 256, 256, 1, 0);

							cn++;
						}
				}

				// back propagation
				for (int n = layers.size() - 1; n > 0; n--) 
				{
					std::string lName = (layers[n])->getLayerName() + "_" + std::to_string((layers[n])->getLayerId());
					if (layers[n]->isBranch())
					{
						std::vector<Reconstruction*>* branch = &(Branches[lName]);;
						for (int br_i = 0; br_i < branch->size(); br_i++)
						{
							Reconstruction* recNet = branch->at(br_i);
							for (int l_j = recNet->layers.size() - 1; l_j > 0; l_j--)
							{
								(recNet->layers[l_j])->backward();
								//(((Branches[lName])[i])->layers[j])->update();

								adam.optimize(recNet->layers[l_j], GLOBAL_ADAMP_ITERA);
								//sgd.optimize(recNet->layers[l_j]);
							}

							// 把每个之路的梯度回传叠加到主路上面
							float a = 1.0;
							// update the convolution filter parameters
							callCuda(cublasSaxpy(cublasHandle, 
								layers[n]->data_size, 
								&a, 
								((recNet->layers[0])->next)->diff, 1,
								layers[n]->next->diff, 1));
						}
					}

					layers[n]->backward();
					//layers[n]->update(); // update the parameters
					adam.optimize(layers[n], GLOBAL_ADAMP_ITERA);
					//sgd.optimize(layers[n]);
				}
				offset += batch;
			}
			GLOBAL_ADAMP_ITERA++;

			/*for (int i = layers.size() - 1; i > 0; i--)
				layers[i]->adjust_learning(step_decrease);*/

			/* reconstruction data and LOSS
			if (size > 0) {
				offset = 0;
				float* batch_loss;
				for (int b = 0; b < size / batch; b++) {
					callCuda(cudaMemcpy(data, h_data + offset * data_dim,
						sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
					callCuda(cudaMemcpy(label, h_label + offset * data_dim,
						sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
					for (int i = 0; i < layers.size(); i++)
						layers[i]->forward(false);
					
					//最后一层时损失层，一般为 Euclidean loss
					//callCuda(cudaMemcpy(batch_loss, layers[layers.size() - 1]->data, 1, cudaMemcpyDeviceToHost));
					loss[0] += (*layers[layers.size() - 1]->data) / batch / (float)data_dim;
					
					int lossN = 1;
					for (std::map<string, std::vector<Reconstruction*>>::iterator bh = Branches.begin(); bh != Branches.end(); ++bh)
						for (Reconstruction* layer_bh : bh->second)
						{
							loss[lossN] += (*layer_bh->layers[layer_bh->layers.size() - 1]->data) / batch / (float)data_dim;
							lossN++;
						}

					if (b==0 && (k) % 50 == 0)
					{
						SaveParams(paras_path);

						// 倒数第二层才是重构层，最后一层时损失层，一般为 Euclidean loss
						callCuda(cudaMemcpy(reconstruction_data + offset * data_dim, layers[layers.size() - 2]->data, 
							sizeof(float) * data_dim * batch, cudaMemcpyDeviceToHost));
						std::string str = "D:/GitHub/VGG_XNet_CUDNN_CUDA/VGG_XNet_CUDNN_CUDA/data/rec_KTD_scarf/branch_" + std::to_string(k) + "_0.bmp";
						net_utils::saveImage(str, reconstruction_data, 256, 256, 1, 0);

						int cn = 1;
						for (std::map<string, std::vector<Reconstruction*>>::iterator bh = Branches.begin(); bh != Branches.end(); ++bh)
							for (Reconstruction* layer_bh : bh->second)
							{
								callCuda(cudaMemcpy(reconstruction_data + offset * data_dim, layer_bh->layers[layer_bh->layers.size() - 2]->data,
									sizeof(float) * data_dim * batch, cudaMemcpyDeviceToHost));
								str = "D:/GitHub/VGG_XNet_CUDNN_CUDA/VGG_XNet_CUDNN_CUDA/data/rec_KTD_scarf/branch_" + std::to_string(k) + "_" + std::to_string(cn) + ".bmp";
								net_utils::saveImage(str, reconstruction_data, 256, 256, 1, 0);
								cn++;
							}
					}

					offset += batch;
				}
			}*/
			for (int ln=0; ln<loss.size(); ln++)
				std::cout << "branch_" <<  ln << " loss: " << loss.at(ln) << " ";
			std::cout << std::endl;

			/*std::string str = "D:/GitHub/VGG_XNet_CUDNN_CUDA/VGG_XNet_CUDNN_CUDA/data/rec_mnist/"+std::to_string(k) + ".png";
			net_utils::saveImage(str, reconstruction_data, 28, 28, 1, 10);*/

			// adjust the learning rate if the validation error stabilizes
			if (k%step_iter == 0)
			{
				adam.adjust_lr(step_decrease);
				std::cout << "Adjust leanrning rate to " << adam.learn_rate << "\n";
			}
		}
	}

	void Reconstruction::Test(std::vector<float*> reconstruction_data) {
		loss = 0;
		int offset = 0;
		float* batch_loss;

		clock_t start1, ends1;
		clock_t start = clock();
		for (int b = 0; b < size / batch; b++) {
			//start1 = clock();
			callCuda(cudaMemcpy(data, h_data + offset * data_dim,
				sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));

				//start1 = clock();
			for (int n = 0; n < layers.size(); n++)
			{
				layers[n]->forward(false);

				std::string lName = (layers[n])->getLayerName() + "_" + std::to_string((layers[n])->getLayerId());
				if (layers[n]->isBranch())
				{
					std::vector<Reconstruction*>* branch = &(Branches[lName]);
					for (int br_i = 0; br_i < branch->size(); br_i++)
					{
						Reconstruction* recNet = branch->at(br_i);
						for (int l_j = 1; l_j < recNet->layers.size(); l_j++)
						{
							(recNet->layers[l_j])->forward(false);
						}
					}
				}
			}
			//ends1 = clock();
			//cout << "Running Foward Time : " << (double)(ends1 - start1) / CLOCKS_PER_SEC * 1000 << "ms" << endl;

			// 倒数第二层才是重构层，最后一层时损失层，一般为 Euclidean loss
			//start1 = clock();

			callCuda(cudaMemcpy(reconstruction_data[0] + offset * data_dim, layers[layers.size() - 2]->data, 
				sizeof(float) * data_dim * batch, cudaMemcpyDeviceToHost));
			int cn = 1;
			for (std::map<string, std::vector<Reconstruction*>>::iterator bh = Branches.begin(); bh != Branches.end(); ++bh)
				for (Reconstruction* layer_bh : bh->second)
				{
					callCuda(cudaMemcpy(reconstruction_data[cn] + offset * data_dim, layer_bh->layers[layer_bh->layers.size() - 2]->data,
						sizeof(float) * data_dim * batch, cudaMemcpyDeviceToHost));
					cn++;
				}

			//ends1 = clock();
			//cout << "Running Reconstruction Data Copy Time : " << (double)(ends1 - start1) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
			offset += batch;
		}
		clock_t ends = clock();
		cout << "Running Time : " << (double)(ends - start) / CLOCKS_PER_SEC * 1000 << "ms" << endl;

		/*std::string str = "D:/GitHub/VGG_XNet_CUDNN_CUDA/VGG_XNet_CUDNN_CUDA/data/rec_lcd_test/" + std::to_string(0) + ".bmp";
		net_utils::saveImage(str, reconstruction_data, 256, 256, 1, 0);*/
		std::cout << "Reconstruction loss: " << loss << std::endl;
	}

	void Reconstruction::PushCluster(int K, float cluster_weight, float alpha, float* init_param, float sigma,
		float momentum, float weight_decay)
	{
		Cluster* cluster = new Cluster(layers.back(), K, cluster_weight, alpha, init_param, sigma, momentum, weight_decay);
		layers.push_back(cluster);
	}

	void Reconstruction::ReadParams(std::string dir) {

		for (int n = 1; n < layers.size() - 1; n++) {
			string name = (layers[n])->getLayerName() + "_" + std::to_string((layers[n])->getLayerId());
			if (!((layers[n])->getLayerName() == "batchnorm") && layers[n]->param_size > 0)
				net_utils::readGPUMatrix(dir + name, layers[n]->param, layers[n]->param_size);

			if (layers[n]->param_bias_size > 0)
				net_utils::readGPUMatrix(dir + name + "_bias", layers[n]->param_bias, layers[n]->param_bias_size);

			if ((layers[n])->getLayerName() == "batchnorm")
			{
				net_utils::readGPUMatrix(dir + name + "_scale", ((BatchNorm*)layers[n])->bnScale, layers[n]->param_size);
				net_utils::readGPUMatrix(dir + name + "_bias", ((BatchNorm*)layers[n])->bnBias, layers[n]->param_size);

				net_utils::readGPUMatrix(dir + name + "_movingMean", ((BatchNorm*)layers[n])->resultMovingMean,
					(((BatchNorm*)layers[n])->data_size) / (((BatchNorm*)layers[n])->batch));
				net_utils::readGPUMatrix(dir + name + "_movingVar", ((BatchNorm*)layers[n])->resultMovingVar,
					(((BatchNorm*)layers[n])->data_size) / (((BatchNorm*)layers[n])->batch));
			}

			if (layers[n]->isBranch())
			{
				std::vector<Reconstruction*> recBranches = Branches[name];
				for (int i = 0; i < recBranches.size(); i++)
					for (int j = 1; j < (recBranches[i])->layers.size() - 1; j++)
					{
						if (!((recBranches[i])->layers[j]->getLayerName() == "batchnorm") && ((recBranches[i])->layers[j])->param_size > 0)
							net_utils::readGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + ((recBranches[i])->layers[j])->getLayerName() + "_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()),
							((recBranches[i])->layers[j])->param, ((recBranches[i])->layers[j])->param_size);

						if (((recBranches[i])->layers[j])->param_bias_size > 0)
							net_utils::readGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + ((recBranches[i])->layers[j])->getLayerName() + "_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_bias",
							((recBranches[i])->layers[j])->param_bias, ((recBranches[i])->layers[j])->param_bias_size);

						if (((recBranches[i])->layers[j])->getLayerName() == "batchnorm")
						{
							net_utils::readGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + "batchnorm_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_scale",
								((BatchNorm*)((recBranches[i])->layers[j]))->bnScale, ((recBranches[i])->layers[j])->param_size);
							net_utils::readGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + "batchnorm_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_bias",
								((BatchNorm*)((recBranches[i])->layers[j]))->bnBias, ((recBranches[i])->layers[j])->param_size);

							net_utils::readGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + "batchnorm_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_movingMean",
								((BatchNorm*)((recBranches[i])->layers[j]))->resultMovingMean, (((recBranches[i])->layers[j])->data_size) / (((recBranches[i])->layers[j])->batch));
							net_utils::readGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + "batchnorm_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_movingVar",
								((BatchNorm*)((recBranches[i])->layers[j]))->resultMovingVar, (((recBranches[i])->layers[j])->data_size) / (((recBranches[i])->layers[j])->batch));
						}
					}
			}
		}
		std::cout << "Params read." << std::endl;
	}

	void Reconstruction::SaveParams(std::string dir) {
		for (int n = 1; n < layers.size() - 1; n++) {
			string name = (layers[n])->getLayerName() + "_" + std::to_string((layers[n])->getLayerId());
			if (!((layers[n])->getLayerName() == "batchnorm") && layers[n]->param_size > 0)
				net_utils::writeGPUMatrix(dir + name, layers[n]->param, layers[n]->param_size);

			if (layers[n]->param_bias_size > 0)
				net_utils::writeGPUMatrix(dir + name + "_bias", layers[n]->param_bias, layers[n]->param_bias_size);
			
			if ((layers[n])->getLayerName() == "batchnorm")
			{
				net_utils::writeGPUMatrix(dir + name + "_scale", ((BatchNorm*)layers[n])->bnScale, layers[n]->param_size);
				net_utils::writeGPUMatrix(dir + name + "_bias", ((BatchNorm*)layers[n])->bnBias, layers[n]->param_size);

				net_utils::writeGPUMatrix(dir + name + "_movingMean", ((BatchNorm*)layers[n])->resultMovingMean, 
					(((BatchNorm*)layers[n])->data_size) / (((BatchNorm*)layers[n])->batch));
				net_utils::writeGPUMatrix(dir + name + "_movingVar", ((BatchNorm*)layers[n])->resultMovingVar,
					(((BatchNorm*)layers[n])->data_size) / (((BatchNorm*)layers[n])->batch));

			}

			if (layers[n]->isBranch())
			{
				std::vector<Reconstruction*> recBranches = Branches[name];
				for (int i = 0; i < recBranches.size(); i++)
					for (int j = 1; j < (recBranches[i])->layers.size() - 1; j++)
					{
						if (!((recBranches[i])->layers[j]->getLayerName() == "batchnorm") && ((recBranches[i])->layers[j])->param_size > 0)
							net_utils::writeGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + ((recBranches[i])->layers[j])->getLayerName() + "_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()),
							((recBranches[i])->layers[j])->param, ((recBranches[i])->layers[j])->param_size);

						if (((recBranches[i])->layers[j])->param_bias_size > 0)
							net_utils::writeGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + ((recBranches[i])->layers[j])->getLayerName() + "_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_bias",
							((recBranches[i])->layers[j])->param_bias, ((recBranches[i])->layers[j])->param_bias_size);

						if (((recBranches[i])->layers[j])->getLayerName() == "batchnorm")
						{
							net_utils::writeGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + "batchnorm_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_scale",
								((BatchNorm*)((recBranches[i])->layers[j]))->bnScale, ((recBranches[i])->layers[j])->param_size);
							net_utils::writeGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + "batchnorm_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_bias",
								((BatchNorm*)((recBranches[i])->layers[j]))->bnBias, ((recBranches[i])->layers[j])->param_size);

							net_utils::writeGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + "batchnorm_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_movingMean",
								((BatchNorm*)((recBranches[i])->layers[j]))->resultMovingMean, (((recBranches[i])->layers[j])->data_size / ((recBranches[i])->layers[j])->batch));
							net_utils::writeGPUMatrix(dir + name + "_branch_" + std::to_string(i) + "_" + "batchnorm_" + std::to_string(((recBranches[i])->layers[j])->getLayerId()) + "_movingVar",
								((BatchNorm*)((recBranches[i])->layers[j]))->resultMovingVar, (((recBranches[i])->layers[j])->data_size) / (((recBranches[i])->layers[j])->batch));
						}
					}
			}
		}
		std::cout << "Params saved." << std::endl;
	}

}