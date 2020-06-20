#include "dataset/mnist.h"
#include "dataset/mnist_reconstruction.h"
#include "dataset/cifar10.h"
#include "dataset/imagenet200.h"
#include "dataset/imagenet.h"

#include "dataset\lcd_reconstruction.h"
#include "dataset\texture_pfean.h"
#include "dataset\lcd_gan.h"

#include "jingce_lcd.h"

#include "layer\dec.h"
#include "utils\set_value.h"
#include "utils\utils.h"
#include "utils\print.h"

using namespace std;

#include "utils\image_priors.h"
void test_prior_extractors()
{
	using namespace net_utils;
	cv::Mat InImg = cv::imread("D:\\Dataset\\texturedefect\\KSRD_fabric\\test\\images\\un_25.bmp", 0);

	cv::Mat PriorImg(256, 256, CV_32FC4);
	float* OutImg = new float[256 * 256 * 1 * 4];
	Prior::extract(InImg.data, OutImg, 3, 256, 256, 1, 1, "");

	std::memcpy(PriorImg.data, OutImg, sizeof(float) * 256 * 256 * 1 * 4);

	delete OutImg;
	std::cout << "test done.\n";
}

int main() {
	cout << "HUSTNet vE309" << endl;
	callCuda(cublasCreate(&global::cublasHandle));
	callCudnn(cudnnCreate(&global::cudnnHandle));

	
	lcd_gan::train();

	//texture::train();

	//test_prior_extractors();

	//lcd_rec::jc_infer();
	//lcd_rec::train();
	//mnist_rec::train();
	//mnist::train();
	//cifar10::train();
	//imagenet200::train();

	/*int feature_dim = 16;
	int feature_num = 64;
	int center_num = 16;*/
	/*int feature_dim = 10;
	int feature_num = 64*64;
	int center_num = 16;
	float* encode_features_ptr = nullptr;
	float* cluster_centers_ptr = nullptr;
	
	float* feature_center_residual_ptr = nullptr;
	float* feature_center_eucild_distance_ptr = nullptr;
	float* source_ptr = nullptr;
	float* target_ptr = nullptr;

	cudaMalloc(&encode_features_ptr, sizeof(float) * feature_dim * feature_num);
	cudaMalloc(&cluster_centers_ptr, sizeof(float) * feature_dim * center_num);

	net_utils::setGpuNormalValue(encode_features_ptr, feature_dim * feature_num, 1, 0.1);
	net_utils::setGpuNormalValue(cluster_centers_ptr, feature_dim * center_num, 1, 0.2);

	net_utils::printGpuMatrix(encode_features_ptr, feature_dim * feature_num, feature_num, feature_dim, 3);
	net_utils::printGpuMatrix(cluster_centers_ptr, feature_dim * center_num, center_num, feature_dim, 3);

	DEC mDec(encode_features_ptr, cluster_centers_ptr, feature_num, center_num, feature_dim, true);
	mDec.clusterLoss_gpu_forward();
	mDec.clusterLoss_gpu_backward();
	
	std::cout << "mDec.mLossToFeaGrad_Dev£º\n";
	net_utils::printGpuMatrix(mDec.mLossToFeaGrad_Dev, feature_dim * feature_num, feature_num, feature_dim, 3);
	std::cout << "mDec.mLossToCenGrad_Dev£º\n";
	net_utils::printGpuMatrix(mDec.mLossToCenGrad_Dev, feature_dim * center_num, center_num, feature_dim, 3);

	std::cout << "mDec.mResidual_Dev£º\n";
	net_utils::printGpuMatrix(mDec.mResidual_Dev, feature_dim * feature_num * center_num, feature_num * center_num, feature_dim, 3);
	std::cout << "mDec.mWeightResidual_Dev 1xKxHWxC £º\n";
	net_utils::printGpuMatrix(mDec.mWeightResidual_Dev, feature_dim * feature_num * center_num, feature_num * center_num, feature_dim, 3);*/

	callCuda(cublasDestroy(global::cublasHandle));
	callCudnn(cudnnDestroy(global::cudnnHandle));
	cout << "End" << endl;
	system("pause");
	return 0;
}



