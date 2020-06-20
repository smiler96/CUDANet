#include "lcd_reconstruction.h"
#include <direct.h>
#include <string>

namespace lcd_rec {

	string mnist_file = global::root + "./data/mnist/";

	int train() {

		string img_name = "1 (35)";
		string modelVersion = "AE";
		string type = "MV_grid";
		string root_path = "D:\\GitHub\\VGG_XNet_CUDNN_CUDA\\VGG_XNet_CUDNN_CUDA\\data\\";
		string paras_path = root_path + type + "\\" + modelVersion + "\\model\\";

		mkdir(paras_path.c_str());
		std::cout << "make folder: \n" << paras_path << "\n";

		//string defect_tpye = "color";
		string test_save_path = root_path + type + "\\" + modelVersion + "\\test\\";
		mkdir(test_save_path.c_str());
		std::cout << "make folder: \n" << test_save_path << "\n";

		string train_images_path = "D:\\Dataset\\texturedefect\\" + type + "\\good_downscale4"; 
		//string train_images_path = root_path + type + "\\train";

		/*string test_images_path = "D:\\Dataset\\¾«²âMuraÍ¼Ïñ\\[1]ÏßÐÔMuraÈ±ÏÝ_2\\" 
			+ modelVersion +  "\\" + img_name + "\\block";*/
		string test_images_path = "D:\\Dataset\\texturedefect\\" + type + "\\test_downscale4\\";
		//string test_images_path = "D:\\Dataset\\texturedefect\\" + type + "\\train_cudnn";

		const int channel = 1;
		const int width = 256, height = 256;
		const int data_dim = channel * width * height;
		int train_size, test_size;

		cout << "Reading input data" << endl;

		// read train data and test data
		std::vector<std::string> train_files, test_files;
		//net_utils::GetAllFiles(train_images_path, train_files);
		net_utils::getAllFormatFiles(train_images_path, train_files, ".png");
		net_utils::getAllFormatFiles(test_images_path, test_files, ".png");
		train_size = train_files.size();
		test_size = test_files.size();

		uint8_t* train_images = new uint8_t[train_size * channel * height * width];
		net_utils::getAllImages(train_images, train_files, data_dim);
		uint8_t* test_images = new uint8_t[test_size * channel * height * width];
		net_utils::getAllImages(test_images, test_files, data_dim);

		// transform data
		//train_size = 1;
		float* h_train_images = new float[train_size * channel * height * width];
		for (int i = 0; i < train_size * channel * height * width; i++)
			h_train_images[i] = (float)train_images[i] / 255.0f;

		//net_utils::showImage(h_train_images, width, height, channel, 0);

		//net_utils::printCpuMatrix(h_train_images, data_dim, width, height, 0);

		int batch_size = 1;
		int iteration = 50000;
		float lr = -1;
		//float moment = 0.9f; // SGD
		float moment = 0.0f; // Adam
		float BNmoment = 0.1f; //BN

		cout << "Batch size: " << batch_size << endl;

		/*model::Reconstruction network(h_train_images, data_dim, h_train_images,
			train_size, 0, batch_size);*/

		float lamta0 = 1.0f, lamta1 = 1.0f, lamta2 = 1.0f;
		std::string InitParaType = "kaiming";

		model::Reconstruction network(train_size, 0, batch_size);
		network.InputTrainData(h_train_images, data_dim, h_train_images, data_dim);
		network.SetLossWeight(lamta0);
		network.PushInput(channel, height, width); // 1 256 256

		/*network.PushConvolution(16, 3, 1, 1, lr, 0.16f, 0.9f, 0.005f); // 8 256 256
		network.PushActivation(CUDNN_ACTIVATION_RELU);
		//network.PushPooling(2, 2); // 8 128 128

		network.PushConvolution(32, 3, 2, 1, lr, 0.12f, 0.9f, 0.005f);  // 8 128 128
		network.PushActivation(CUDNN_ACTIVATION_RELU);
		//network.PushPooling(2, 2);  // 16 64 64

		network.PushConvolution(64, 3, 2, 1, lr, 0.13f, 0.9f, 0.005f); // 16 64 64
		network.PushActivation(CUDNN_ACTIVATION_RELU);
		//network.PushPooling(2, 2);  // 64 32 32

		network.PushConvolution(128, 3, 2, 1, lr, 0.15f, 0.9f, 0.005f); // 16 32 32
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushConvolution(10, 3, 2, 1, lr, 0.1f, 0.9f, 0.005f); // 10 16 16
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(128, 3, 2, 1, lr, 0.13f, 0.9f, 0.005f); // 16 32 32
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(64, 3, 2, 1, lr, 0.2f, 0.9f, 0.005f); // 16 64 64
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(32, 3, 2, 1, lr, 0.11f, 0.9f, 0.005f); // 8 128 128
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(16, 3, 2, 1, lr, 0.15f, 0.9f, 0.005f); // 8 256 256
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(1, 3, 1, 1, lr, 0.09f, 0.9f, 0.005f); // 1 256 256
		network.PushActivation(CUDNN_ACTIVATION_SIGMOID);*/

		network.PushConvolution(16, 3, 1, 1, lr, 0.015f, moment, 0.005f); // 32 256 256
		network.PushBatchNorm(16, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);
		//network.PushPooling(2, 2); // 8 128 128

		network.PushConvolution(32, 3, 2, 1, lr, 0.05f, moment, 0.005f);  // 64 128 128
		network.PushBatchNorm(32, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);
		//network.PushPooling(2, 2);  // 16 64 64

		network.PushConvolution(64, 3, 2, 1, lr, 0.02f, moment, 0.005f); // 128 64 64
		network.PushBatchNorm(64, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);
			//network.PushBranchNet(network.layers.back(), network.label);
			//std::string key1 = network.layers.back()->getLayerName() + "_" + std::to_string(network.layers.back()->getLayerId());
			//model::Reconstruction* branch1 = (network.Branches[key1]).back();
			//branch1->SetLossWeight(lamta1);

			//// Òþ¿Õ¼ä
			//branch1->PushConvolution(10, 3, 2, 1, lr, 0.09f, 0.9f, 0.005f); // 10 32 32
			//branch1->PushBatchNorm(10, 1e-5, BNmoment, lr, 0.11f, 0.9f, 0.00f);
			//branch1->PushActivation(CUDNN_ACTIVATION_RELU);

			//branch1->PushDeconvolution(64, 3, 2, 1, lr, 0.23f, 0.9f, 0.005f); // 128 64 64
			//branch1->PushBatchNorm(64, 1e-5, BNmoment, lr, 0.11f, 0.9f, 0.00f);
			//branch1->PushActivation(CUDNN_ACTIVATION_RELU);

			//branch1->PushDeconvolution(32, 3, 2, 1, lr, 0.15f, 0.9f, 0.005f); // 64 128 128
			//branch1->PushBatchNorm(32, 1e-5, BNmoment, lr, 0.11f, 0.9f, 0.00f);
			//branch1->PushActivation(CUDNN_ACTIVATION_RELU);

			//branch1->PushDeconvolution(16, 3, 2, 1, lr, 0.15f, 0.9f, 0.005f); // 32 256 256
			//branch1->PushBatchNorm(16, 1e-5, BNmoment, lr, 0.11f, 0.9f, 0.00f);
			//branch1->PushActivation(CUDNN_ACTIVATION_RELU);

			//branch1->PushDeconvolution(1, 3, 1, 1, lr, 0.11f, 0.9f, 0.005f); // 1 256 256
			//branch1->PushActivation(CUDNN_ACTIVATION_SIGMOID);

			//branch1->PushEuclideanLoss();
			//branch1->InitParas(InitParaType);

		network.PushConvolution(128, 3, 2, 1, lr, 0.013f, moment, 0.005f); // 256 32 32
		network.PushBatchNorm(128, 1e-5, BNmoment, lr, 0.09f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);
		//network.PushCluster(16, 0.1, lr, NULL);
			//network.PushBranchNet(network.layers.back(), network.label);
			//std::string key0 = network.layers.back()->getLayerName() + "_" + std::to_string(network.layers.back()->getLayerId());
			//model::Reconstruction* branch0 = (network.Branches[key0]).back();
			//branch0->SetLossWeight(lamta2);

			//// Òþ¿Õ¼ä
			//branch0->PushConvolution(10, 3, 2, 1, lr, 0.09f, 0.9f, 0.005f); // 10 16 16
			//branch0->PushBatchNorm(10, 1e-5, BNmoment, lr, 0.11f, 0.9f, 0.00f);
			//branch0->PushActivation(CUDNN_ACTIVATION_RELU);

			//branch0->PushDeconvolution(128, 3, 2, 1, lr, 0.03f, 0.9f, 0.005f); // 256 32 32
			//branch0->PushBatchNorm(128, 1e-5, BNmoment, lr, 0.11f, 0.9f, 0.00f);
			//branch0->PushActivation(CUDNN_ACTIVATION_RELU);

			//branch0->PushDeconvolution(64, 3, 2, 1, lr, 0.23f, 0.9f, 0.005f); // 128 64 64
			//branch0->PushBatchNorm(64, 1e-5, BNmoment, lr, 0.11f, 0.9f, 0.00f);
			//branch0->PushActivation(CUDNN_ACTIVATION_RELU);

			//branch0->PushDeconvolution(32, 3, 2, 1, lr, 0.15f, 0.9f, 0.005f); // 64 128 128
			//branch0->PushBatchNorm(32, 1e-5, BNmoment, lr, 0.11f, 0.9f, 0.00f);
			//branch0->PushActivation(CUDNN_ACTIVATION_RELU);

			//branch0->PushDeconvolution(16, 3, 2, 1, lr, 0.15f, 0.9f, 0.005f); // 32 256 256
			//branch0->PushBatchNorm(16, 1e-5, BNmoment, lr, 0.11f, 0.9f, 0.00f);
			//branch0->PushActivation(CUDNN_ACTIVATION_RELU);

			//branch0->PushDeconvolution(1, 3, 1, 1, lr, 0.11f, 0.9f, 0.005f); // 1 256 256
			//branch0->PushActivation(CUDNN_ACTIVATION_SIGMOID);

			//branch0->PushEuclideanLoss();
			//branch0->InitParas(InitParaType); 


		network.PushConvolution(256, 3, 2, 1, lr, 0.103f, moment, 0.005f); // 256 16 16
		network.PushBatchNorm(256, 1e-5, BNmoment, lr, 0.09f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		// Òþ¿Õ¼ä
		network.PushConvolution(10, 3, 2, 1, lr, 0.01f, moment, 0.005f); // 10 8 8
		network.PushBatchNorm(10, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(256, 3, 2, 1, lr, 0.09f, moment, 0.005f); // 256 16 16
		network.PushBatchNorm(256, 1e-5, BNmoment, lr, 0.11f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(128, 3, 2, 1, lr, 0.019f, moment, 0.005f); // 256 32 32
		network.PushBatchNorm(128, 1e-5, BNmoment, lr, 0.0101f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(64, 3, 2, 1, lr, 0.105f, moment, 0.005f); // 128 64 64
		network.PushBatchNorm(64, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(32, 3, 2, 1, lr, 0.05f, moment, 0.005f); // 64 128 128
		network.PushBatchNorm(32, 1e-5, BNmoment, lr, 0.15f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(16, 3, 2, 1, lr, 0.15f, moment, 0.005f); // 32 256 256
		network.PushBatchNorm(16, 1e-5, BNmoment, lr, 0.105f, moment, 0.00f);
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(1, 3, 1, 1, lr, 0.05f, moment, 0.005f); // 1 256 256
		network.PushActivation(CUDNN_ACTIVATION_SIGMOID);

		network.PushEuclideanLoss();
		network.InitParas(InitParaType);

		network.PrintGeneral();

		// train the model
		std::cout << "Networks has: " << network.Branches.size() << " branches.\n";

		cout << "Train " << iteration << " times ..." << endl;
		//network.ReadParams(paras_path);
		network.Train(paras_path, iteration, 800, 0.1, false); // depend on the number of batch_size
		cout << "End of training ..." << endl;

		//network.SaveParams(paras_path);

		// transform test data
		float* h_test_images = new float[test_size * channel * height * width];
		for (int i = 0; i < test_size * channel * height * width; i++)
			h_test_images[i] = (float)test_images[i] / 255.0f;

		// test the model
		network.SwitchData(h_test_images, h_test_images, test_size);
		//network.SwitchData(h_test_images, h_test_images, 468);

		cout << "Testing ..." << endl;
		std::vector<float*> h_test_rec(network.Branches.size() + 1);
		for (int i=0; i<h_test_rec.size(); i++)
			h_test_rec[i] = new float[test_size * channel * height * width];

		network.Test(h_test_rec);
		cout << "End of testing ..." << endl;
		for (int offset=0; offset<test_size; offset++)
		{
			std::string str = test_save_path + "\\" + std::to_string(offset) + "_init.bmp";
			net_utils::saveImage(str, h_test_images, 256, 256, 1, offset);

			for (int i = 0; i < h_test_rec.size(); i++)
			{
				str = test_save_path + "\\" + std::to_string(offset) + "_branch_" + std::to_string(i) + ".bmp";
				/*string name = test_files[offset].substr(test_images_path.size() + 1);
				std::string str ="D:\\Dataset\\¾«²âMuraÍ¼Ïñ\\[1]ÏßÐÔMuraÈ±ÏÝ_2\\" + modelVersion + "\\" + img_name + "\\infer\\" + name;*/
				net_utils::saveImage(str, h_test_rec[i], 256, 256, 1, offset);
			}

		}

		for (int i = 0; i<h_test_rec.size(); i++)
			delete[] h_test_rec[i];
		delete[] test_images;
		delete[] h_test_images;

		delete[] train_images;
		delete[] h_train_images;

		return 0;
	}

}