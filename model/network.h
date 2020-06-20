/*
 * network.h
 *
 * This class uses all layer class to construct a neural network, user should
 * have his data and label array in the host machine (and test data eventually).
 * By calling the constructor and layer inserter, user can build his own network
 * with adjusted parameters like layer size, learning rate, etc..
 *
 * The learning method is gradient descent and the learning rate decreases when
 * validation error stabilizes.
 *
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include <iostream>
#include <vector>
#include <map>
#include <typeinfo>
#include <cmath>

#include "../layer/layer.h"
#include "../layer/input.h"
#include "../layer/output.h"
#include "../layer/convolution.h"
#include "../layer/deconvolution.h"
#include "../layer/pooling.h"
#include "../layer/fully_connected.h"
#include "../layer/activation.h"
#include "../layer/relu.h"
#include "../layer/softmax.h"
#include "../layer/softmax_activationnloss.h"
#include "../layer/euclidean_loss.h"
#include "../layer/softmax_loss.h"
#include "../layer/batch_normalization.h"

#include "../init/init.h"
#include "../init/kaiming_init.h"
#include "../init/xavier_init.h"

#include "../utils/read_data.h"
#include "../utils/write_data.h"

using namespace layer;

namespace model {

class Network {
public:
	std::vector<Layer*> layers; // list of layers
	float* data; // input on device, = train_data + val_data
	float* h_data; // input on host
	int data_dim; // dimension of one input, c x h x w
	float* label; // label on device (gpu)
	float* h_label; // label on host (cpu)
	//int label_dim; // dimension of one label (usually 1)
	int size, val_size, batch; // whole size of train_data, val_data, batch size

	float lambda; // cumulative learning rate adjustment
	float loss_weight;
	
	int convNum = 0; // 卷积层个数
	int poolNum = 0; // 池化层个数
	int deconvNum = 0; //反卷积层个数	
	int fullyconnectedNum = 0; // 全连接层个数
	int activateNum = 0; // 激活层个数
	int bnNum = 0; // batch normalization 层个数
	
	//std::vector<Network*> LayerBranches; // layer层的分支
	std::map<std::string, std::vector<Network*>> Branches; // 所有layer层的分支
	
	/*
	 * Constructor
	 *
	 * data: pointer to host data array (include training and validation data)
	 * data_dim: dimension of one single datum
	 * label: pointer to host label array
	 * label_dim: dimension of label, usually 1
	 * train_size: training data count
	 * val_size: validation data count
	 * batch_size: batch size, common for all layers
	 */
	 Network(float* data, int data_dim, float* label, /*int label_dim,*/
			 int train_size, int val_size, int batch_size);
	
	Network(int _batch);
	Network(int _train_size, int _val_size, int _batch);
	
	// Network(Layer* layer);
	void PushBranchNet(Layer* layer, float* _label_dev);
	
	void InputTrainData(float* _data, int _data_dim, float* _label, int _label_diml);
	
	virtual ~Network();

	/*
	 * Get data and label pointer in order to change it from the outside
	 */
	std::pair<float*, float*> GetData();

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
	/*virtual void Train(int iteration, float half_time = 0, float half_rate = 0.5,
		float step_decrease = 1, bool debug = false) = 0;*/

	/*
	* 设置损失权重
	*
	* _loss_weight: s损失权重
	*/
	void SetLossWeight(const float _loss_weight);

	/*
	 * Insert input layer (should be called at first)
	 *
	 * c: channel number
	 * h: height
	 * w: width
	 */
	void PushInput(int c, int h, int w);

	/*
	 * Insert output layer (should be called in the end)
	 *
	 * label_count: label dimension (example: 10 for digits)
	 */
	/*void PushOutput(int label_count);*/

	/*
	 * Insert convolutional layer without activation layer (mode NCHW)
	 *
	 * c: channel of the layer
	 * kernel: square kernel size
	 * stride: square stride for convolution operation
	 * padding: square padding for convolution operation
	 * alpha: initial learning rate
	 * sigma: initial weight distribution
	 * momentum: momentum of gradient when learning
	 * weight_decay: decay rate of parameters
	 */
	void PushConvolution(int c, int kernel, int stride, int padding, float alpha, float sigma = 0.01f,
			float momentum = 0.9f, float weight_decay = 0);

	/*
	* Insert deconvolutional layer without activation layer (mode NCHW)
	*
	* c: channel of the layer
	* kernel: square kernel size
	* stride: square stride for deconvolution operation
	* padding: square padding for deconvolution operation
	* alpha: initial learning rate
	* sigma: initial weight distribution
	* momentum: momentum of gradient when learning
	* weight_decay: decay rate of parameters
	*/
	void PushDeconvolution(int c, int kernel, int stride, int padding, float alpha, float sigma = 0.01f,
		float momentum = 0.9f, float weight_decay = 0);

	/*
	 * Insert max pooling layer
	 *
	 * size: size of the square pool
	 * stride: overlapping
	 */
	void PushPooling(int size, int stride);

	/*
	* Insert fully connected layer
	*
	* LJY
	* 2020.1.17
	*/
	void PushFullyConnected(int output_size, float dropout_rate, float alpha,
		float sigma, float momentum, float weight_decay);

	/*
	 * Insert activation layer (after convolutional layer)
	 *
	 * mode: activation function (CuDNN constant)
	 */
	void PushActivation(cudnnActivationMode_t mode);

	/*
	 * Insert fully-connected layer with ReLU as activation function
	 *
	 * output_size: output size of the current layer
	 * dropout_rate: rate of dropout
	 * alpha: initial learning rate
	 * sigma: initial weight distribution
	 * momentum: momentum of gradient when learning
	 * weight_decay: decay rate of parameters
	 */
	void PushReLU(int output_size, float dropout_rate, float alpha,
			float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);

	/*
	 * Insert fully-connected layer with Softmax as activation function
	 *
	 * output_size: output size of the current layer
	 * dropout_rate: rate of dropout
	 * alpha: initial learning rate
	 * sigma: initial weight distribution
	 * momentum: momentum of gradient when learning
	 * weight_decay: decay rate of parameters
	 */
	/*void PushSoftmax(int output_size, float dropout_rate, float alpha,
			float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);*/

	/*
	* Insert softmax and cross entropy loss layer
	*
	* LJY
	* 2020.1.17
	*/
	void PushSoftmaxAnL(int _class_num);

	/*
	* Insert euclidean loss layer in the end
	*/
	void PushEuclideanLoss();

	/*
	* Insert softmax loss layer in the end of softmax layer
	*/
	//void PushSoftmaxLoss(int _class_num, int _batch, float* _label);
	void PushSoftmaxLoss(int _class_num);

	/*
	* Insert batch normalization layer after conv
	*/
	void PushBatchNorm(int _channels, float _epsilon, float _expAverFactor, float alpha, float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);

	/*
	 * Remove last layer
	 */
	void Pop();

	/*
	 * Switch the host data, used when data size is huge or when testing
	 *
	 * h_data: pointer to new data array
	 * h_label: pointer to new label array
	 * size: data size
	 */
	void SwitchData(float* h_data, float* h_label, int size);

	void SwitchDataD(float* data, float* label, int count);

	/*
	 * Test the network, used after switching the test data
	 *
	 * h_label: label array to store the prediction
	 */
	/*virtual void Test(float* h_label) = 0;*/

	/*
	 * Print general information about layers
	 */
	void PrintGeneral();

	/*
	 * Print a datum
	 *
	 * offset: position of datum
	 * r: row
	 * c: column
	 */
	void PrintData(int offset, int r, int c, int precision);

	/*
	 * Get parameters from files (one folder with multiple files), the actual
	 * network structure should match the saved one
	 *
	 * dir: path to the folder
	 */
	void ReadParams(std::string dir);

	/*
	 * Save parameters to files (one folder with multiple files)
	 *
	 * dir: path to the folder
	 */
	void SaveParams(std::string dir);

	void InitParas(std::string InitType);
};

} /* namespace model */
#endif /* NETWORK_H_ */
