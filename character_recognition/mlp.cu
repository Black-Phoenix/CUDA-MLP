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

	void printCuda(float *a1, int n, string name) {
		float *print_a = new float[n];
		cout << name.c_str() << endl;
		cout << "{" << endl;
		cudaMemcpy(print_a, a1, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			cout << "\t" << print_a[i] << endl;
		}
		cout << "}" << endl;
		delete[]print_a;
	}

	__global__ void bias_addition(int n, float *A, float *B, float *C) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		C[index] = A[index] + B[index];
	}

	__global__ void relu_activation(int n, float *A, float *C) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		C[index] = max(0.0f, A[index]);
	}

	__global__ void softmax_activation(int n, float *A, float *C, float exp_sum) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		C[index] = expf(A[index]) / exp_sum;
	}


	__global__ void scan(int n, float *data, int d) {// function to get sum (for softmax layer)
		int tmp_d = 1 << (d + 1);
		int index = (blockDim.x * blockIdx.x + threadIdx.x)*tmp_d;
		if (index >= n)
			return;
		data[index + tmp_d - 1] += data[index + (tmp_d >> 1) - 1];
	}

	__global__ void exp_copy(int n, float *odata, float *idata) {// kernal to copy exp(idata[i]) to odata[i]
		int index = (blockDim.x * blockIdx.x + threadIdx.x);
		if (index >= n)
			return;
		odata[index] = expf(idata[index]);
	}

	__global__ void gpu_matrix_mult(float *a, float *b, float *c, int m, int n, int k)
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
	__global__ void fill_data(int n, float *data, float val) {
		int index = (blockDim.x * blockIdx.x + threadIdx.x);
		if (index >= n)
			return;
		data[index] = val;
	}

	void Net::GPU_fill_rand(float *A, int size, float std) {
		// Create a pseudo-random number generator
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

		// Set the seed for the random number generator using the system clock
		curandSetPseudoRandomGeneratorSeed(prng, clock());

		// Fill the array with random numbers on the device
		curandGenerateNormal(prng, A, size, 0, std);
	}


	Net::Net(int n, vector<int> layers) : input_size(n), layer_sizes(layers) {
		// layers = {98, 52, 52}
		layers.insert(layers.begin(), n);
		float *dev_w, *dev_b, *dev_g, *dev_a;
		int blocks;
		for (int i = 0; i < layers.size() - 1; i++) {
			cudaMalloc((void**)&dev_w, (layers[i] * layers[i + 1]) * sizeof(float));
			cudaMalloc((void**)&dev_b, (layers[i + 1]) * sizeof(float));
			// initilize w, b using gaussian distribution
			GPU_fill_rand(dev_w, layers[i] * layers[i + 1], 2.0 / layers[i]); // henormal initilization
			printCuda(dev_w, layers[i] * layers[i + 1], "W fresh");
			GPU_fill_rand(dev_b, layers[i + 1], 0.1f); // zero initilizaton is fine for biases
			// push into vector
			w.push_back(dev_w);
			b.push_back(dev_b);
			// int results
			cudaMalloc((void**)&dev_g, (layers[i + 1]) * sizeof(float));
			cudaMalloc((void**)&dev_a, (layers[i + 1]) * sizeof(float));
			g.push_back(dev_g);
			a.push_back(dev_a);
		}
		// initilizaton cublas handle
		cublasCreate(&handle);
	}

	// C(m,n) = A(m,k) * B(k,n)
	void Net::gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
		int lda = m, ldb = k, ldc = m;
		const float alf = 1;
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		// Do the actual multiplication
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	

	float* Net::forward(float *data, int n) {
		float *res = new float[classes]();
		assert(n == input_size);
		float *dev_data;
		cudaMalloc((void**)&dev_data, n * sizeof(float));
		cudaMemcpy(dev_data, data, n * sizeof(float), cudaMemcpyHostToDevice);
		for (int i = 0; i < layer_sizes.size(); i++) {
			blocks = ceil((layer_sizes[i] + block_size - 1) / block_size);
			// clear g, a for this layer
			cudaMemset(g[i], 0, layer_sizes[i] * sizeof(float));
			checkCUDAErrorWithLine("Cuda memset failed!");
			cudaMemset(a[i], 0, layer_sizes[i] * sizeof(float));
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
			if (i != layer_sizes.size() - 1) {
				// relu activation
				relu_activation << <blocks, block_size >> > (layer_sizes[i], g[i], a[i]);
				checkCUDAErrorWithLine("relu failed!");
			}
			else {
				exp_copy << <blocks, block_size >> > (layer_sizes[i], a[i], g[i]);
				checkCUDAErrorWithLine("exp copy failed!");
				// todo optimize this later
				float *tmp = new float[layer_sizes[i]];
				float exp_sum = 0;
				cudaMemcpy(tmp, a[i], layer_sizes[i] * sizeof(float), cudaMemcpyDeviceToHost);
				for (int pos = 0; pos < layer_sizes[i]; pos++)
					exp_sum += tmp[pos];
				delete[] tmp;
				// modified scan to get the exponential sum of all elements (P1 of assignment used!!)
				//int closest_pow2 = 1 << ilog2ceil(layer_sizes[i]);
				//int blocks_scan;
				//for (int d = 0; d <= ilog2ceil(closest_pow2) - 1; d++) {
				//	// compute number of threads to spawn
				//	blocks_scan = ceil((closest_pow2 / (1 << (d + 1)) + block_size - 1) / block_size);
				//	scan<<<blocks_scan, block_size>>>(closest_pow2, a[i], d);
				//	checkCUDAErrorWithLine("scan failed!");
				//}
				//cudaMemcpy(&exp_sum, a[i] + closest_pow2 - 1, sizeof(float), cudaMemcpyDeviceToHost);
				// softmax activation
				checkCUDAErrorWithLine("Cuda memcpy failed!");
				softmax_activation << <blocks, block_size >> > (layer_sizes[i], g[i], a[i], exp_sum);
				checkCUDAErrorWithLine("softmax failed!");
			}
		}
		cudaMemcpy(res, a[layer_sizes.size() - 1], classes * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorWithLine("Cuda res memcpy failed!");
		return res;
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
		// clean culbas hand
		cublasDestroy(handle);
	}
}
