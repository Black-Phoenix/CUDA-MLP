#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>
#include <memory>
#include <string>
#include <iostream>

// kernals
namespace CharacterRecognition {
#define enable_debug true
	using Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	void printCuda(double *a1, int n, string name) {
		if (!enable_debug)
			return;
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
	
	__global__ void relu_grad(int n, double *z, double * grad) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		grad[index] = (z[index] > 0) ? 1 : 0;; // makes a diagona matrix
	}

	__global__ void softmax_activation(int n, double *A, double *C, double exp_sum) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n)
			return;
		C[index] = exp(A[index]) / exp_sum;
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

	__global__ void element_mult(int n, double *a, double *b, double *c) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= n)
			return;
		c[index] = a[index] * b[index];
	}

	__global__ void update_params(int n, double *param, double *grad, double lr) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= n)
			return;
		param[index] -= lr * grad[index];
	}

	__global__ void memset(int n, double *data, float value) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= n)
			return;
		data[index] = value;
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

	Net::Net(int n, vector<int> layers, double lr) {
		// layers = {98, 52, 52}
		params.layer_count = layers.size();
		params.input_size = n;
		params.lr = lr;
		params.layer_sizes = layers;
		// init raw data holder
		cudaMalloc((void**)&dev_data, n * sizeof(double));
		cudaMalloc((void**)&dev_y, layers[params.layer_count - 1] * sizeof(double));
		layers.insert(layers.begin(), n);
		double *dev_w, *dev_b, *dev_z, *dev_a, *dev_da;
		int blocks;
		for (int i = 0; i < params.layer_count; i++) {
			cudaMalloc((void**)&dev_w, (layers[i] * layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			cudaMalloc((void**)&dev_b, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			// initilize w, b using gaussian distribution
			GPU_fill_rand(dev_w, layers[i] * layers[i + 1], 2.0 / layers[i]); // uniform random initilization
			// memset dev_b
			blocks = (layers[i + 1] + params.block_size - 1) / params.block_size;
			memset << <blocks, params.block_size >> > (layers[i + 1], dev_b, 0.1);
			//GPU_fill_rand(dev_b, layers[i + 1], 2.0 / layers[i]); // zero initilizaton is fine for biases
			// push into vector
			w.push_back(dev_w);
			b.push_back(dev_b);
			// intermediate results arrays
			cudaMalloc((void**)&dev_z, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			cudaMalloc((void**)&dev_a, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			z.push_back(dev_z);
			a.push_back(dev_a);

			// grad arrays
			cudaMalloc((void**)&dev_w, (layers[i] * layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			dL_dw.push_back(dev_w); // gradient of w

			cudaMalloc((void**)&dev_da, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			dL_dz.push_back(dev_da); // da/dg has dimensions output(g) * output(g) <Jacobian>

			cudaMalloc((void**)&dev_da, (layers[i]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			dL_da.push_back(dev_da); // da/dg has dimensions output(g) * output(g) <Jacobian>
			// relu grad
			cudaMalloc((void**)&dev_z, (layers[i + 1]) * sizeof(double));
			checkCUDAErrorWithLine("Cuda malloc failed!");
			d_relu.push_back(dev_z);
		}
		// initilizaton cublas handle
		cublasCreate(&handle); 
		/*for (int i = 0; i < params.layer_count; i++) {
			printCuda(w[i], (layers[i] * layers[i + 1]), "INIT W" + to_string(i));
			printCuda(b[i], (layers[i + 1]), "INIT B" + to_string(i));
		}*/
	}

	// C(m,n) = A(m,k) * B(k,n)
	// lda = k (if transposed)
	// ldb = n (if we transpose)
	// ldb = n (if we transpose)
	void Net::gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, bool trans_flag_a, bool trans_flag_b) {
		int lda, ldb, ldc;
		lda = (!trans_flag_a) ? m : k;
		ldb = (!trans_flag_b) ? k : n;
		ldc = m;
		const double alf = 1; // gpu vs cpu
		const double bet = 0;
		const double *alpha = &alf;
		const double *beta = &bet;
		// Do the actual multiplication
		cublasDgemm(handle, (cublasOperation_t)trans_flag_a, (cublasOperation_t)trans_flag_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	double* Net::forward(double *data, int n) {
		double *res = new double[params.layer_sizes[params.layer_count - 1]]();
		assert(n == params.input_size);
		// copy over data to process
		cudaMemcpy(dev_data, data, n * sizeof(double), cudaMemcpyHostToDevice);
		checkCUDAErrorWithLine("Cuda memcpy failed!");
		for (int i = 0; i < params.layer_count; i++) {
			blocks = ceil((params.layer_sizes[i] + params.block_size - 1) / params.block_size);
			// clear g, a for this layer
			cudaMemset(z[i], 0, params.layer_sizes[i] * sizeof(double));
			checkCUDAErrorWithLine("Cuda memset failed!");
			cudaMemset(a[i], 0, params.layer_sizes[i] * sizeof(double));
			checkCUDAErrorWithLine("Cuda memset failed!");
			// matrix multiplication
			if (!i) { // first iteration, so a[i-1] hasn't been set yet
				gpu_blas_mmul(w[i], dev_data, z[i], params.layer_sizes[i], params.input_size, 1); // wx + b
				checkCUDAErrorWithLine("gpu mult failed!");
			}
			else {
				gpu_blas_mmul(w[i], a[i - 1], z[i], params.layer_sizes[i], params.layer_sizes[i - 1], 1);// wx + b
				checkCUDAErrorWithLine("gpu mult failed!");
			}
			// bias addition
			bias_addition << <blocks, params.block_size >> > (params.layer_sizes[i], z[i], b[i], z[i]); // put result back into z[i]
			checkCUDAErrorWithLine("bias addition failed!");
			if (i != params.layer_count - 1) {
				// relu activation
				relu_activation << <blocks, params.block_size >> > (params.layer_sizes[i], z[i], a[i]);
				checkCUDAErrorWithLine("relu failed!");
			}
			else {
				//exp_copy << <blocks, params.block_size >> > (params.layer_sizes[i], a[i], z[i]); // why do this anymore lol
				checkCUDAErrorWithLine("exp copy failed!");
				// todo move this to the gpu this later
				double *tmp = new double[params.layer_sizes[i]];
				double exp_sum = 0;
				cudaMemcpy(tmp, z[i], params.layer_sizes[i] * sizeof(double), cudaMemcpyDeviceToHost); // used to be a[i]
				checkCUDAErrorWithLine("Cuda memcpy failed!");
				for (int pos = 0; pos < params.layer_sizes[i]; pos++)
					exp_sum += exp(tmp[pos]);
				delete[] tmp;
				// modified scan to get the exponential sum of all elements (P1 of assignment used!!)
				// softmax activation
				checkCUDAErrorWithLine("Cuda memcpy failed!");
				softmax_activation <<<blocks, params.block_size >>> (params.layer_sizes[i], z[i], a[i], exp_sum);
				checkCUDAErrorWithLine("softmax failed!");
			}
		}
		cudaMemcpy(res, a[params.layer_count - 1], params.layer_sizes[params.layer_count - 1] * sizeof(double), cudaMemcpyDeviceToHost);
		checkCUDAErrorWithLine("Cuda res memcpy failed!");
		return res;
	}

	void Net::backprop(double *y) {
		// calculate loss grad
		int n = -1;
		cudaMemcpy(dev_y, y, params.layer_sizes[params.layer_count - 1] * sizeof(double), cudaMemcpyHostToDevice);
		checkCUDAErrorWithLine("Cuda memcpy failed!");
		for (int i = params.layer_count - 1; i >= 0; i--) {
			n = params.layer_sizes[i];
			blocks = ceil((n + params.block_size - 1) / params.block_size);
			if (i == params.layer_count - 1) { // softmax
				//printCuda(a[i], n, "A" + to_string(i));
				bias_addition << <blocks, params.block_size >> > (n, a[i], dev_y, dL_dz[i], -1); // y_hat - y
				checkCUDAErrorWithLine("bias addition failed!");
			}
			else { // relu 
				//printCuda(z[i], n, "z"+to_string(i));
				relu_grad<< <blocks, params.block_size >> > (n, z[i], d_relu[i]); // relu grad
				checkCUDAErrorWithLine("relu grad failed!");
				/*printCuda(d_relu[i], n, "d_relu" + to_string(i));
				printCuda(dL_da[i], n, "dA" + to_string(i));*/
				element_mult << <blocks, params.block_size >> > (n, d_relu[i], dL_da[i], dL_dz[i]); // dz = da * a'
				checkCUDAErrorWithLine("element wise mult failed!");
			}
			if (i != 0) {
				//printCuda(dL_dz[i], params.layer_sizes[i], "dL_dz" + to_string(i));
				gpu_blas_mmul(dL_dz[i], a[i-1], dL_dw[i], params.layer_sizes[i], 1, params.layer_sizes[i-1], false, true); // dw
				checkCUDAErrorWithLine("Matrix mult failed!");
				//printCuda(dL_dw[i], params.layer_sizes[i] * params.layer_sizes[i - 1], "dL_dw" + to_string(i));
				//printCuda(dL_da[i - 1], params.layer_sizes[i - 2], "dA" + to_string(i - 1) + " Prv loop");
				gpu_blas_mmul(w[i], dL_dz[i], dL_da[i - 1], params.layer_sizes[i - 1], params.layer_sizes[i], 1, true, false); // da
				checkCUDAErrorWithLine("Matrix mult failed!");
				//printCuda(dL_da[i-1], params.layer_sizes[i-2], "dA" + to_string(i - 1) + " Prv loop");
				checkCUDAErrorWithLine(("Print error failed! " + to_string(i)).c_str());
			}
			else {
				gpu_blas_mmul(dL_dz[i], dev_data, dL_dw[i], params.layer_sizes[i], 1, params.input_size, false, true); // dw
				checkCUDAErrorWithLine("Matrix mult failed!");
			}
		}
		for (int i = 0; i < params.layer_count; i++) {
			int layer_im1;
			if (i != 0)
				layer_im1 = params.layer_sizes[i - 1];
			else 
				layer_im1 = params.input_size;
			// W
			n = params.layer_sizes[i] * layer_im1;
			blocks = ceil((n + params.block_size - 1) / params.block_size);
			update_params << <blocks, params.block_size >> > (n, w[i], dL_dw[i], params.lr);
			checkCUDAErrorWithLine("Update w failed!");
			//B
			n = params.layer_sizes[i];
			blocks = ceil((n + params.block_size - 1) / params.block_size);
			update_params << <blocks, params.block_size >> > (n, b[i], dL_dz[i], params.lr);
			checkCUDAErrorWithLine("Update b failed!");
			// print grads
			//printCuda(dL_dw[i], layer_im1 * params.layer_sizes[i], "Grad W" + to_string(i));
			//printCuda(dL_dz[i], params.layer_sizes[i], "Grad b"+to_string(i));
			checkCUDAErrorWithLine("Print failed!");
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
		for (auto x : z)
			cudaFree(x);
		for (auto x : a)
			cudaFree(x);
		// grads
		for (auto x : dL_dz)
			cudaFree(x);
		for (auto x : dL_da)
			cudaFree(x);
		for (auto x : dL_dw)
			cudaFree(x);
		for (auto x : d_relu)
			cudaFree(x);
		cudaFree(dev_data);
		cudaFree(dev_y);
		// clean culbas hand
		cublasDestroy(handle);
	}
}
