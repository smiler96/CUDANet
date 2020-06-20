#include "common_funcs_gpu.h"

namespace func_gpu {


	__global__ void _inverse(const float* __restrict__ x, int n, float alpha, float* __restrict__ y)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n)
		{
			y[index] = alpha / x[index];
		}
	}

	__global__ void _pow(const float* __restrict__ x, int n, int p, float alpha, float b, float* __restrict__ y)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n)
		{
			y[index] = alpha * pow(x[index], p) + b;
		}

	}

	__global__ void _log(const float* __restrict__ x, int n, float alpha, float b, float* __restrict__  y)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n)
		{
			y[index] = alpha * log(x[index]) + b;
		}
	}

	__global__ void _process_element(const float* __restrict__  vector_ptr, int n, float alpha, float b, float* __restrict__  vector_out_ptr)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n)
		{
			vector_out_ptr[index] = alpha * vector_ptr[index] + b;
		}
	}

	__global__ void _vector_add(int n, const float* __restrict__ x, float a, const float* __restrict__ y, float b, float c, float* __restrict__ z)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n)
		{ 
			z[index] = a * x[index] + b * y[index] + c;
		}

	}

	__global__ void _vector_divide_element(const float* __restrict__ x, const float* __restrict__ y, int n, float alpha, float b, float* __restrict__ z)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n)
		{
			z[index] = alpha * x[index] / y[index] + b;
		}
	}

	__global__ void _vector_multiply_element(const float* __restrict__ x, const float* __restrict__ y, int n, float alpha, float b, float* __restrict__ z)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n)
		{
			z[index] = alpha * x[index] * y[index] + b;
		}
	}


	__global__ void  _divide_element_by_row(const float* __restrict__ x, int n, const float* __restrict__ y, int rows, int clos, float* __restrict__ z)
	{
		int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标(fea)
		int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标(cen)
		int global_id = y_id * clos + x_id; // 总坐标(residual)

		if (x_id < clos && y_id < rows)
		{
			*(z + global_id) = *(x + global_id) / *(y + x_id);
		}
	}

	__global__ void  _divide_element_by_col(const float* __restrict__ x, int n, const float* __restrict__ y, int rows, int clos, float* __restrict__ z)
	{
		int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标(fea)
		int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标(cen)
		int global_id = y_id * clos + x_id; // 总坐标(residual)

		if (x_id < clos && y_id < rows)
		{
			*(z + global_id) = *(x + global_id) / *(y + y_id);
		}
	}

	__global__ void _asy_multiply_element(const float* __restrict__ x, int xN, const float* __restrict__ y, int yN, int step, float* z)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < xN)
		{
			z[index] = x[index] * y[index / step];
		}
	}



	__device__ inline static float EuclidDistance(const float* __restrict__ objects, const float* __restrict__ clusters, int objLength)
	{
		float dist = 0.0f;

		for (int i = 0; i < objLength; i++)
		{
			float onePoint = objects[i] - clusters[i];
			dist = onePoint * onePoint + dist;
		}

		return(dist);
	}

	__global__ void _encode_residual(int fDim, const float* __restrict__ feature, int feaNum, const float* __restrict__ center, int cenNum, float* __restrict__ residual)
	{
		int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标(fea)
		int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标(cen)
		int global_id = y_id * feaNum * fDim + x_id; // 总坐标(residual)

		if (x_id < feaNum * fDim && y_id < cenNum)
		{
			*(residual + global_id) = *(feature + x_id) - *(center + y_id * fDim + x_id % fDim);
		}
	}

	__global__ void _encode_distance(int fDim, const float* __restrict__ feature, int feaNum, const float* __restrict__ center, int cenNum, float* __restrict__ distance)
	{
		int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标(fea)
		int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标(cen)
		if (x_id < feaNum && y_id < cenNum)
		{
			*(distance + y_id *feaNum + x_id) = EuclidDistance(feature + x_id * fDim, center + y_id * fDim, fDim);
		}
	}

	__global__ void _diverge_KL(const float* __restrict__ x, const float* __restrict__ y, int n, float* z)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n)
		{
			z[index] = x[index] * log(x[index] / y[index]);
		}
	}

	void inverseGpu(const float* __restrict__ x, int n, float alpha, float* __restrict__ y)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (n + threadsPerblock - 1) / threadsPerblock;
		_inverse << < blocksPerGrid, threadsPerblock >> > (x, n, alpha, y);
	}

	void logGpu(const float* __restrict__ x, int n, float alpha, float b, float* __restrict__  y)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (n + threadsPerblock - 1) / threadsPerblock;
		_log << < blocksPerGrid, threadsPerblock >> > (x, n, alpha, b, y);
	}

	void powGpu(const float* __restrict__ x, int n, int p, float alpha, float b, float* __restrict__  y)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (n + threadsPerblock - 1) / threadsPerblock;
		_pow << < blocksPerGrid, threadsPerblock >> > (x, n, p, alpha, b, y);
	}

	void processElementGpu(const float* __restrict__ x, int n, float alpha, float b, float* __restrict__  y)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (n + threadsPerblock - 1) / threadsPerblock;
		_process_element << < blocksPerGrid, threadsPerblock >> > (x, n, alpha, b, y);
	}

	void vectorAddGpu(int n, const float* __restrict__ x, float a, const float* __restrict__ y, float b, float c, float* __restrict__ z)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (n + threadsPerblock - 1) / threadsPerblock;
		_vector_add << < blocksPerGrid, threadsPerblock >> > (n, x, a, y, b, c, z);
	}

	void vectordivideElemGpu(const float* __restrict__ x, const float* __restrict__ y, int n, float alpha, float b, float* __restrict__ z)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (n + threadsPerblock - 1) / threadsPerblock;
		_vector_divide_element << < blocksPerGrid, threadsPerblock >> > (x, y, n, alpha, b, z);
	}

	void vectorMultiplyElemGpu(const float* __restrict__ x, const float* __restrict__ y, int n, float alpha, float b, float* __restrict__ z)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (n + threadsPerblock - 1) / threadsPerblock;
		_vector_multiply_element << < blocksPerGrid, threadsPerblock >> > (x, y, n, alpha, b, z);
	}

	void  divideElembyRowGpu(const float* __restrict__ x, int n, const float* __restrict__ y, int rows, int clos, float* __restrict__ z)
	{
		dim3 dimBlock2D(ThreadsPerBlock_16, ThreadsPerBlock_16);
		dim3 dimGrid2D((clos + ThreadsPerBlock_16 - 1) / ThreadsPerBlock_16, (rows + ThreadsPerBlock_16 - 1) / ThreadsPerBlock_16);
		_divide_element_by_row << < dimGrid2D, dimBlock2D >> > (x, n, y, rows, clos, z);
	}

	void  divideElembyColGpu(const float* __restrict__ x, int n, const float* __restrict__ y, int rows, int clos, float* __restrict__ z)
	{
		dim3 dimBlock2D(ThreadsPerBlock_16, ThreadsPerBlock_16);
		dim3 dimGrid2D((clos + ThreadsPerBlock_16 - 1) / ThreadsPerBlock_16, (clos + ThreadsPerBlock_16 - 1) / ThreadsPerBlock_16);
		_divide_element_by_col << < dimGrid2D, dimBlock2D >> > (x, n, y, rows, clos, z);
	}

	void asyMultiplyElemGpu(const float* __restrict__ x, int xN, const float* __restrict__ y, int yN, int step, float* z)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (xN + threadsPerblock - 1) / threadsPerblock;
		_asy_multiply_element << < blocksPerGrid, threadsPerblock >> > (x, xN, y, yN, step, z);
	}


	void encodeResidualGpu(int fDim, const float* __restrict__ feature, int feaNum, const float* __restrict__ center, int cenNum, float* __restrict__ residual)
	{
		dim3 dimBlock2D(ThreadsPerBlock_16, ThreadsPerBlock_16);
		dim3 dimGrid2D((fDim * feaNum + ThreadsPerBlock_16 - 1) / ThreadsPerBlock_16, (cenNum + ThreadsPerBlock_16 - 1) / ThreadsPerBlock_16);
		_encode_residual << < dimGrid2D, dimBlock2D >> > (fDim, feature, feaNum, center, cenNum, residual);
	}

	void encodeDistanceGpu(int fDim, const float* __restrict__ feature, int feaNum, const float* __restrict__ center, int cenNum, float* __restrict__ distance)
	{
		dim3 dimBlock2D(ThreadsPerBlock_16, ThreadsPerBlock_16);
		dim3 dimGrid2D((feaNum + ThreadsPerBlock_16 - 1) / ThreadsPerBlock_16, (cenNum + ThreadsPerBlock_16 - 1) / ThreadsPerBlock_16);
		_encode_distance << < dimGrid2D, dimBlock2D >> > (fDim, feature, feaNum, center, cenNum, distance);

	}

	void divergKLGpu(const float* __restrict__ x, const float* __restrict__ y, int n, float* z)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (n + threadsPerblock - 1) / threadsPerblock;
		_diverge_KL << < blocksPerGrid, threadsPerblock >> > (x, y, n, z);
	}



	__global__ void _adam_update_g(const float* __restrict__ m1, const float* __restrict__ m2, const int n, const int iter, const float beta1, const float beta2, const float epsilon, float* __restrict__ gradient_update)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n)
		{
			float d_m1 = m1[index] / (float)(1 - pow(beta1, iter));
			float d_m2 = m2[index] / (float)(1 - pow(beta2, iter));
			gradient_update[index] = d_m1 / (sqrt(d_m2) + epsilon);
		}

	}

	void adamUpdateGpu(const float* m1, const float* m2, const int n, const int iter, const float beta1, const float beta2, const float epsilon, float* gradient_update)
	{
		int threadsPerblock = ThreadsPerBlock_16 * ThreadsPerBlock_16;
		int blocksPerGrid = (n + threadsPerblock - 1) / threadsPerblock;
		_adam_update_g << < blocksPerGrid, threadsPerblock >> > (m1, m2, n, iter, beta1, beta2, epsilon, gradient_update);
	}
}