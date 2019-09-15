#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>
#include <memory>
#include <iostream>

// kernals
namespace CharacterRecognition {
	using Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	void printCuda(double *a1, int n, string name) {
		double *print_a = new double[n];
		cout << name.c_str() << endl;
		cout << "{" << endl;
		cudaMemcpy(print_a, a1, n * sizeof(double), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			cout << "\t" << print_a[i] << endl;
		}
		cout << "}" << endl;
		delete[]print_a;
	}
	//////////////////////////////
	/*			KERNALS			*/	
	//////////////////////////////
	__global__ void bias_addition(int n, double *A, double *B, double *C, int sign = 1) { // change sign for subtraction or scaled addition
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		C[index] = A[index] + sign*B[index];
	}

	__global__ void relu_activation(int n, double *A, double *C) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		C[index] = max(0.0f, A[index]);
	}
	
	__global__ void relu_grad(int n, double *g, double * grad) { // assumes grad is a 2d array of size n x n
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		grad[index * n + index] = max(0.0f, g[index]); // makes a diagona matrix
	}

	__global__ void softmax_activation(int n, double *A, double *C, double exp_sum) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		C[index] = exp(A[index]) / exp_sum;
	}

	__global__ void softmax_grad(int n, double *g, double * grad) {
		int index_i = threadIdx.x + (blockIdx.x * blockDim.x);
		int index_j = threadIdx.y + (blockIdx.y * blockDim.y);
		if (index_i >= n || index_j >= n)
			return;
		grad[index_i * n + index_j] = g[index_i] * ((index_i == index_j) - g[index_j]);
	}

	__global__ void scan(int n, double *data, int d) {// function to get sum (for softmax layer)
		int tmp_d = 1 << (d + 1);
		int index = (blockDim.x * blockIdx.x + threadIdx.x)*tmp_d;
		if (index >= n)
			return;
		data[index + tmp_d - 1] += data[index + (tmp_d >> 1) - 1];
	}

	__global__ void exp_copy(int n, double *odata, double *idata) {// kernal to copy exp(idata[i]) to odata[i]
		int index = (blockDim.x * blockIdx.x + threadIdx.x);
		if (index >= n)
			return;
		odata[index] = exp(idata[index]);
	}

	__global__ void fill_data(int n, double *data, double val) {
		int index = (blockDim.x * blockIdx.x + threadIdx.x);
		if (index >= n)
			return;
		data[index] = val;
	}

	__global__ void gpu_matrix_mult(const double *a, const double *b, double *c, const int m, const int n, const int k)
	{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int sum = 0; 
		if (col < k && row < m)
		{
			for (int i = 0; i < n; i++)
			{
				sum += a[row * n + i] * b[i * k + col];
			}
			c[row * k + col] = sum;
		}
	}

	//////////////////////////////
	/*			Helper			*/
	//////////////////////////////
	
	void Net::GPU_fill_rand(double *A, int size, double std) {
		// Create a pseudo-random number generator
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

		// Set the seed for the random number generator using the system clock
		curandSetPseudoRandomGeneratorSeed(prng, clock());

		// Fill the array with random numbers on the device
		curandGenerateNormalDouble(prng, A, size, 0, std);
	}

	Net::Net(int n, vector<int> layers) : input_size(n), layer_sizes(layers) {
		// layers = {98, 52, 52}
		// init raw data holder
		cudaMalloc((void**)&dev_data, n * sizeof(double));
		layer_count = layers.size();
		layers.insert(layers.begin(), n);
		double *dev_w, *dev_b, *dev_g, *dev_a, *dev_jac;
		int blocks;
		for (int i = 0; i < layer_count; i++) {
			cudaMalloc((void**)&dev_w, (layers[i] * layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			cudaMalloc((void**)&dev_b, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			// initilize w, b using gaussian distribution
			GPU_fill_rand(dev_w, layers[i] * layers[i + 1], 2.0 / layers[i]); // uniform random initilization
			//printCuda(dev_w, layers[i] * layers[i + 1], "W fresh");
			GPU_fill_rand(dev_b, layers[i + 1], 0.1f); // zero initilizaton is fine for biases
			// push into vector
			w.push_back(dev_w);
			b.push_back(dev_b);
			// intermediate results arrays
			cudaMalloc((void**)&dev_g, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			cudaMalloc((void**)&dev_a, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			g.push_back(dev_g);
			a.push_back(dev_a);
			// grad arrays
			cudaMalloc((void**)&dev_w, (layers[i] * layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			dL_dw.push_back(dev_w);
			cudaMalloc((void**)&dev_b, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			dL_db.push_back(dev_b);
			cudaMalloc((void**)&dev_jac, (layers[i + 1]) * (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			da_dg.push_back(dev_jac); // da/dg has dimensions output(g) * output(g) <Jacobian>
		}
		// initilizaton cublas handle
		cublasCreate(&handle);
		// init final grad buffer
		cudaMalloc((void**)&dL_dyhat, layers[layer_count] * sizeof(double));
	}

	// C(m,n) = A(m,k) * B(k,n)
	void Net::gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
		int lda = m, ldb = k, ldc = m;
		const double alf = 1;
		const double bet = 0;
		const double *alpha = &alf;
		const double *beta = &bet;
		// Do the actual multiplication
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	double* Net::forward(double *data, int n) {
		double *res = new double[layer_sizes[layer_count - 1]]();
		assert(n == input_size);
		cudaMemcpy(dev_data, data, n * sizeof(double), cudaMemcpyHostToDevice);
		for (int i = 0; i < layer_count; i++) {
			blocks = ceil((layer_sizes[i] + block_size - 1) / block_size);
			// clear g, a for this layer
			cudaMemset(g[i], 0, layer_sizes[i] * sizeof(double));
			checkCUDAErrorWithLine("Cuda memset failed!");
			cudaMemset(a[i], 0, layer_sizes[i] * sizeof(double));
			checkCUDAErrorWithLine("Cuda memset failed!");
			int block_size_tmp = 16;
			int col_blocks = (1 + block_size_tmp - 1) / block_size_tmp, row_col = (layer_sizes[i]+ block_size_tmp - 1)/ block_size_tmp;
			// matrix multiplication
			if (!i) { // first iteration, so a[i] hasn't been set yet
				gpu_blas_mmul(dev_data, w[i], g[i], 1, input_size, layer_sizes[i]);
				checkCUDAErrorWithLine("gpu mult failed!");
			}
			else {
				gpu_blas_mmul(a[i - 1], w[i], g[i], 1, layer_sizes[i-1], layer_sizes[i]);
				checkCUDAErrorWithLine("gpu mult failed!");
			}
			// bias addition
			bias_addition << <blocks, block_size >> > (layer_sizes[i], g[i], b[i], g[i]); // put result back into g[i]
			checkCUDAErrorWithLine("bias addition failed!");
			if (i != layer_count - 1) {
				// relu activation
				relu_activation << <blocks, block_size >> > (layer_sizes[i], g[i], a[i]);
				checkCUDAErrorWithLine("relu failed!");
			}
			else {
				exp_copy << <blocks, block_size >> > (layer_sizes[i], a[i], g[i]);
				checkCUDAErrorWithLine("exp copy failed!");
				// todo move this to the gpu this later
				double *tmp = new double[layer_sizes[i]];
				double exp_sum = 0;
				cudaMemcpy(tmp, a[i], layer_sizes[i] * sizeof(double), cudaMemcpyDeviceToHost);
				for (int pos = 0; pos < layer_sizes[i]; pos++)
					exp_sum += tmp[pos];
				delete[] tmp;
				// modified scan to get the exponential sum of all elements (P1 of assignment used!!)
				// softmax activation
				checkCUDAErrorWithLine("Cuda memcpy failed!");
				softmax_activation << <blocks, block_size >> > (layer_sizes[i], g[i], a[i], exp_sum);
				checkCUDAErrorWithLine("softmax failed!");
			}
		}
		cudaMemcpy(res, a[layer_count - 1], layer_sizes[layer_count - 1] * sizeof(double), cudaMemcpyDeviceToHost);
		checkCUDAErrorWithLine("Cuda res memcpy failed!");
		return res;
	}

	void Net::backprop(double *y) {
		// calculate loss grad
		blocks = ceil((layer_sizes[layer_count - 1] + block_size - 1) / block_size);
		bias_addition <<<blocks, block_size >> > (layer_sizes[layer_count - 1], a[layer_count - 1], y, dL_dyhat, -1);
		// call softmax with correct number of threads
		for (int i = layer_count - 1; i >= 0; i--) {
			blocks = ceil((layer_sizes[i] + block_size - 1) / block_size);
			if (i == layer_count - 1) { // softmax grad
				dim3 gridDim(blocks, blocks);
				dim3 blockDim(block_size, block_size);
				softmax_grad <<< gridDim, blockDim >>> (layer_sizes[i], g[i], da_dg[i]);
			}
			else { // relu grad
				relu_grad <<< blocks, block_size >>> (layer_sizes[i], g[i], da_dg[i]);
			}
			// 
		}
	}

	double Net::loss(int *y) {
		return -1;
	}
	Net::~Net() {
		// free weights and biases
		for (auto x : w)
			cudaFree(x);
		for (auto x : b)
			cudaFree(x);
		// intermediate values
		for (auto x : g)
			cudaFree(x);
		for (auto x : a)
			cudaFree(x);
		// grads
		for (auto x : dL_dw) 
			cudaFree(x);
		for (auto x : dL_db)
			cudaFree(x);
		for (auto x : da_dg)
			cudaFree(x);
		cudaFree(dL_dyhat);
		cudaFree(dev_data);
		// clean culbas hand
		cublasDestroy(handle);
	}
}
