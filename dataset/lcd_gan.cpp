#include "lcd_gan.h"
#include <direct.h>
#include <string>

namespace lcd_gan {

	int train() {
		string train_images_path = "D:\\Dataset\\texturedefect\\MV_carpet\\good_downscale4\\";
		string test_images_path = train_images_path;
		string paras_path = " ";

		const int channel = 1;
		const int width = 256, height = 256;
		const int data_dim = channel * width * height;
		const int class_num = 2;
		int train_size, test_size;

		cout << "Reading input data" << endl;

		// read train data and test data
		std::vector<std::string> train_files, test_files;
		//net_utils::GetAllFiles(train_images_path, train_files);
		net_utils::getAllFormatFiles(train_images_path, train_files, ".png");
		net_utils::getAllFormatFiles(test_images_path, test_files, ".bmp");
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

		int batch_size = 32;
		int iteration = 50000;
		float lr = -1;
		//float moment = 0.9f; // SGD
		float moment = 0.0f; // Adam
		float BNmoment = 0.1f; //BN

		cout << "Batch size: " << batch_size << endl;

		/*model::Reconstruction network(h_train_images, data_dim, h_train_images,
		train_size, 0, batch_size);*/

		float lamta0 = 1.0f, lamta1 = 1.0f;
		std::string InitParaType = "kaiming";


		model::GAN network(h_train_images, data_dim, train_size, batch_size);
		
		// discrimintor ÍøÂç
		network.discrimintor->PushInput(channel, height, width);
		network.discrimintor->PushConvolution(4, 3, 2, 1, lr, 0.015f, moment, 0.005f); // 4 128 128
		network.discrimintor->PushBatchNorm(4, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.discrimintor->PushActivation(CUDNN_ACTIVATION_RELU);

		network.discrimintor->PushConvolution(8, 3, 2, 1, lr, 0.05f, moment, 0.005f);  // 8 64 64
		network.discrimintor->PushBatchNorm(8, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.discrimintor->PushActivation(CUDNN_ACTIVATION_RELU);

		network.discrimintor->PushConvolution(8, 3, 2, 1, lr, 0.02f, moment, 0.005f); // 8 32 32
		network.discrimintor->PushBatchNorm(8, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.discrimintor->PushActivation(CUDNN_ACTIVATION_RELU);

		network.discrimintor->PushFullyConnected(512, 0, lr, 0.011f, 0.9f, 0.005f);
		network.discrimintor->PushActivation(CUDNN_ACTIVATION_RELU);
		network.discrimintor->PushFullyConnected(2, 0, lr, 0.011f, 0.9f, 0.005f);
		network.discrimintor->PushSoftmaxAnL(2);

		// generator ÍøÂç
		network.generator->SetLossWeight(lamta0);
		network.generator->PushInput(channel, height, width); // 1 256 256

		network.generator->PushConvolution(16, 3, 1, 1, lr, 0.015f, moment, 0.005f); // 32 256 256
		network.generator->PushBatchNorm(16, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushConvolution(32, 3, 2, 1, lr, 0.05f, moment, 0.005f);  // 64 128 128
		network.generator->PushBatchNorm(32, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushConvolution(64, 3, 2, 1, lr, 0.02f, moment, 0.005f); // 128 64 64
		network.generator->PushBatchNorm(64, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushConvolution(128, 3, 2, 1, lr, 0.013f, moment, 0.005f); // 256 32 32
		network.generator->PushBatchNorm(128, 1e-5, BNmoment, lr, 0.09f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushConvolution(256, 3, 2, 1, lr, 0.103f, moment, 0.005f); // 256 16 16
		network.generator->PushBatchNorm(256, 1e-5, BNmoment, lr, 0.09f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushConvolution(10, 3, 2, 1, lr, 0.01f, moment, 0.005f); // 10 8 8 Òþ¿Õ¼ä
		network.generator->PushBatchNorm(10, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushDeconvolution(256, 3, 2, 1, lr, 0.09f, moment, 0.005f); // 256 16 16
		network.generator->PushBatchNorm(256, 1e-5, BNmoment, lr, 0.11f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushDeconvolution(128, 3, 2, 1, lr, 0.019f, moment, 0.005f); // 256 32 32
		network.generator->PushBatchNorm(128, 1e-5, BNmoment, lr, 0.0101f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushDeconvolution(64, 3, 2, 1, lr, 0.105f, moment, 0.005f); // 128 64 64
		network.generator->PushBatchNorm(64, 1e-5, BNmoment, lr, 0.015f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushDeconvolution(32, 3, 2, 1, lr, 0.05f, moment, 0.005f); // 64 128 128
		network.generator->PushBatchNorm(32, 1e-5, BNmoment, lr, 0.15f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushDeconvolution(16, 3, 2, 1, lr, 0.15f, moment, 0.005f); // 32 256 256
		network.generator->PushBatchNorm(16, 1e-5, BNmoment, lr, 0.105f, moment, 0.00f);
		network.generator->PushActivation(CUDNN_ACTIVATION_RELU);

		network.generator->PushDeconvolution(1, 3, 1, 1, lr, 0.05f, moment, 0.005f); // 1 256 256
		network.generator->PushActivation(CUDNN_ACTIVATION_SIGMOID);

		network.generator->PushEuclideanLoss();
		network.generator->InitParas(InitParaType);



		// train the model
		cout << "Train " << iteration << " times ..." << endl;
		//network.ReadParas(paras_path);
		network.Train(paras_path, iteration, 0.1, true);
		cout << "End of training ..." << endl;
		//network.SaveParams(paras_path);

		//for (int offset = 0; offset<test_size; offset++)
		//{
		//	std::string str = test_save_path + "\\" + std::to_string(offset) + "_init.bmp";
		//	net_utils::saveImage(str, h_test_images, 256, 256, 1, offset);

		//	for (int i = 0; i < h_test_rec.size(); i++)
		//	{
		//		str = test_save_path + "\\" + std::to_string(offset) + "_branch_" + std::to_string(i) + ".bmp";
		//		/*string name = test_files[offset].substr(test_images_path.size() + 1);
		//		std::string str ="D:\\Dataset\\¾«²âMuraÍ¼Ïñ\\[1]ÏßÐÔMuraÈ±ÏÝ_2\\" + modelVersion + "\\" + img_name + "\\infer\\" + name;*/
		//		net_utils::saveImage(str, h_test_rec[i], 256, 256, 1, offset);
		//	}

		//}

		/*for (int i = 0; i < h_test_rec.size(); i++)
			delete[] h_test_rec[i];*/
		delete[] test_images;
		/*delete[] h_test_images;*/

		delete[] train_images;
		delete[] h_train_images;

		return 0;
	}

}