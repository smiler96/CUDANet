#include "layer.h"

namespace layer {

Layer::Layer(float alpha, float momentum, float weight_decay):
		alpha(alpha), momentum(momentum), weight_decay(weight_decay) {
	
	layerName = "";
	layerId = -1;
	branch = false;

	data = nullptr;  
	data_size = 0; 
	diff = nullptr;  
	param = nullptr;
	param_size = 0; 
	param_bias = nullptr;  
	param_bias_size = 0; 
	gradient = nullptr; 
	gradient_bias = nullptr; 
	batch = 0; 

	para_moment1 = nullptr; // first moment vector
	para_moment2 = nullptr; // second moment vector

	bias_moment1 = nullptr; // first moment vector
	bias_moment2 = nullptr; // second moment vector
	
}


void Layer::setLayerName(std::string _name)
{
	layerName = _name;
}

void Layer::setLayerId(int _id)
{
	layerId = _id;
}

std::string Layer::getLayerName()
{
	return layerName;
}

int Layer::getLayerId()
{
	return layerId;
}

bool Layer::isBranch()
{
	return branch;
}

void Layer::setBranch()
{
	branch = true;
}

Layer::~Layer() 
{
	callCuda(cudaFree(para_moment1));
	callCuda(cudaFree(para_moment2));
	callCuda(cudaFree(gradient2));
	callCuda(cudaFree(gradient3));

	callCuda(cudaFree(bias_moment1));
	callCuda(cudaFree(bias_moment2));
	callCuda(cudaFree(bias_gradient2));
	callCuda(cudaFree(bias_gradient3));

}

void Layer::adjust_learning(float scale) {
	alpha *= scale;
}


}

