#include "texture_pfean.h"

namespace texture {

	int train()
	{
		
		string modelVersion = "PFEAN_V1";
		string defect = "MV_grid";
		string train_images_path = "D:\\Dataset\\texturedefect\\" + defect + "\\good_downscale4\\";
		string test_images_path = "D:\\Dataset\\texturedefect\\" + defect + "\\test_downscale4\\";
		
		string save_root = "C:\\Users\\ms952\\Desktop\\";
		string paras_path = save_root + modelVersion + "\\" + defect + "\\model\\";
		string test_save_path = save_root + modelVersion + "\\" + defect + "\\test\\";
		string train_prior_path = save_root + modelVersion + "\\" + defect + "\\train_prior\\";
		string test_prior_path = save_root + modelVersion + "\\" + defect + "\\test_prior\\";
		mkdir(paras_path.c_str());
		std::cout << "make folder: \n" << paras_path << "\n";
		mkdir(test_save_path.c_str());
		std::cout << "make folder: \n" << test_save_path << "\n";

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

		// 提取训练集先验信息
		uint8_t* train_images = new uint8_t[train_size * channel * height * width];
		net_utils::getAllImages(train_images, train_files, data_dim);
		float* h_train_in_data = new float[train_size * data_dim * (PRIOR_NUM + 1)];
		if (access((train_prior_path + "prior").c_str(), 0) != -1)
		{
			net_utils::readCPUMatrix(train_prior_path + "prior", h_train_in_data, train_size * data_dim * (PRIOR_NUM + 1));
		}
		else
		{
			PFEAN::PriorExtractor::extract(train_images, h_train_in_data, PRIOR_NUM, height, width, channel, train_size, train_prior_path);
			net_utils::writeCPUMatrix(train_prior_path + "prior", h_train_in_data, train_size * data_dim * (PRIOR_NUM + 1));
		}

		// 提取测试集先验信息
		uint8_t* test_images = new uint8_t[test_size * channel * height * width];
		net_utils::getAllImages(test_images, test_files, data_dim);
		float* h_test_in_data = new float[test_size * data_dim * (PRIOR_NUM + 1)];
		if (access((test_prior_path + "prior").c_str(), 0) != -1)
		{
			net_utils::readCPUMatrix(test_prior_path + "prior", h_test_in_data, train_size * data_dim * (PRIOR_NUM + 1));
		}
		else
		{
			PFEAN::PriorExtractor::extract(test_images, h_test_in_data, PRIOR_NUM, height, width, channel, test_size, test_prior_path);
			net_utils::writeCPUMatrix(test_prior_path + "prior", h_test_in_data, test_size * data_dim * (PRIOR_NUM + 1));
		}

		net_utils::showImage(h_test_in_data, 256, 256, 1, 0);
	}

}