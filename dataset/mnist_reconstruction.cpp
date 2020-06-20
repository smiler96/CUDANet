#include "mnist_reconstruction.h"

namespace mnist_rec {

	const int channel = 1;

	string mnist_file = global::root + "./data/mnist/";

	int train() {
		string train_images_path = mnist_file + "train-images.idx3-ubyte";
		string test_images_path = mnist_file + "t10k-images.idx3-ubyte";

		int width = 28, height = 28;
		int train_size, test_size;

		cout << "Reading input data" << endl;

		// read train data
		ifstream train_images_file(train_images_path, ios::binary);
		train_images_file.seekg(4);
		net_utils::readInt(train_images_file, &train_size);
		net_utils::readInt(train_images_file, &height);
		net_utils::readInt(train_images_file, &width);
		uint8_t* train_images = new uint8_t[train_size * channel * height * width];
		net_utils::readBytes(train_images_file, train_images,
			train_size * channel * height * width);
		train_images_file.close();

		int data_dim = channel * width * height;

		cout << "Done. Training dataset size: " << train_size << endl;
		// transform data
		float* h_train_images = new float[train_size * channel * height * width];
		//float* h_train_labels = new float[train_size];
		
		for (int i = 0; i < train_size * channel * height * width; i++)
			h_train_images[i] = (float)train_images[i] / 255.0f;
		net_utils::printCpuMatrix(h_train_images, data_dim, 28, 28, 0);

		int val_size = 0;
		train_size -= val_size; // 50000

								// build LeNet

		int batch_size = 100;
		//int batch_size = 1;
		int iteration = 200;
		float lr = -0.1;

		cout << "Batch size: " << batch_size << endl;

		model::Reconstruction network(h_train_images, data_dim, h_train_images,
			train_size, val_size, batch_size);
		network.PushInput(channel, height, width); // 1 28 28

		network.PushConvolution(8, 3, 1, 0, lr, 0.1f, 0.9f, 0.005f); // 32 14 14
		network.PushActivation(CUDNN_ACTIVATION_RELU); 
		//network.PushPooling(2, 2);  // 32 14 14

		network.PushConvolution(16, 3, 1, 0, lr, 0.12f, 0.9f, 0.005f);  // 64 7 7
		network.PushActivation(CUDNN_ACTIVATION_RELU);
		////network.PushPooling(2, 2);  // 64 7 7

		network.PushConvolution(32, 3, 1, 0, lr, 0.15f, 0.9f, 0.005f); // 128 7 7
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		//network.PushReLU(800, 0.6, -8e-2f, 0.02f, 0.15f, 0.005f);
		//network.PushSoftmax(label_count, 0.25, -8e-2f, 0.015f, 0.9f, 0.005f);
		//network.PushOutput(label_count);

		network.PushDeconvolution(16, 3, 1, 0, lr, 0.16f, 0.9f, 0.005f); // 64 14 14
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(8, 3, 1, 0, lr, 0.14f, 0.9f, 0.005f); // 32 28 28
		network.PushActivation(CUDNN_ACTIVATION_RELU);

		network.PushDeconvolution(1, 3, 1, 0, lr, 0.11f, 0.9f, 0.005f); // 1 28 28
		network.PushActivation(CUDNN_ACTIVATION_SIGMOID);

		network.PushEuclideanLoss();

		network.PrintGeneral();

		// train the model

		cout << "Train " << iteration << " times ..." << endl;
		//network.ReadParams(mnist_file);
		network.Train(mnist_file, iteration, iteration, 0.1, true); // depend on the number of batch_size
		cout << "End of training ..." << endl;

		network.SaveParams(mnist_file);

		// read test data
		cout << "Reading test data" << endl;
		ifstream test_images_file(test_images_path, ios::binary);
		test_images_file.seekg(4);
		net_utils::readInt(test_images_file, &test_size);
		net_utils::readInt(test_images_file, &height);
		net_utils::readInt(test_images_file, &width);
		uint8_t* test_images = new uint8_t[test_size * channel * height * width];
		test_images_file.read((char*)test_images, test_size * channel * height * width);
		test_images_file.close();

		cout << "Done. Test dataset size: " << test_size << endl;
		// transform test data
		float* h_test_images = new float[test_size * channel * height * width];
		/*float* h_test_labels = new float[test_size];*/
		for (int i = 0; i < test_size * channel * height * width; i++)
			h_test_images[i] = (float)test_images[i] / 255.0f;

		// test the model

		network.SwitchData(h_test_images, h_test_images, test_size);

		cout << "Testing ..." << endl;
		/*float* h_test_labels_predict = new float[test_size * channel * height * width];
		network.Test(h_test_labels_predict);*/
		cout << "End of testing ..." << endl;
		//vector<int> errors;
		//for (int i = 0; i < test_size; i++) {
		//	if (abs(h_test_labels_predict[i] - h_test_labels[i]) > 0.1) {
		//		errors.push_back(i);
		//		//cout << h_test_labels_predict[i] << ' ' << h_test_labels[i] << endl;
		//	}
		//}
		//cout << "Error rate: " << (0.0 + errors.size()) / test_size * 100 << endl;

		//delete[] h_test_labels_predict;
		delete[] test_images;
		delete[] h_test_images;

		delete[] train_images;
		delete[] h_train_images;

		return 0;
	}

}
