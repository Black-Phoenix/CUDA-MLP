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

	__global__ void update_momentum(int n, double *vdw, double *dL_dw, double beta) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= n)
			return;
		vdw[index] = beta * vdw[index] + (1 - beta) * dL_dw[index];
	}

	__global__ void cross_entropy_kernal(int n, double *y, double *y_hat, double *dev_loss) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= n)
			return;
		dev_loss[index] = y[index] * log2(y_hat[index]);
	}

	/*From Stream compaction part of this assignment*/
	__global__ void reduce_kern(int n, double *data, int d) {
		int tmp_d = 1 << (d + 1);
		int index = (blockDim.x * blockIdx.x + threadIdx.x)*tmp_d;
		if (index >= n)
			return;
		double tmp_data = data[index + (tmp_d >> 1) - 1]; // saves a read or write
		if (tmp_data == 0)
			return;
		data[index + tmp_d - 1] += tmp_data;
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

	Net::Net(int n, vector<int> layers, double lr, double beta) {
		// layers = {98, 52, 52}
		params.layer_count = layers.size();
		params.input_size = n;
		params.output_size = layers[params.layer_count - 1];
		params.lr = lr;
		params.layer_sizes = layers;
		if (beta != -1) {
			momentum_grad = true;
			params.beta = beta;
		}
		else
			momentum_grad = false;
		// init raw data holder
		cudaMalloc((void**)&dev_data, n * sizeof(double));
		cudaMalloc((void**)&dev_y, params.output_size * sizeof(double));
		cudaMalloc((void**)&dev_reduction_pow2, 1<<(ilog2ceil(params.output_size)) * sizeof(double));
		// add input layer to front
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
			checkCUDAErrorWithLine("Memset failed!");
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
			// momentum variables
			if (momentum_grad) {
				// Vb
				cudaMalloc((void**)&dev_z, (layers[i + 1]) * sizeof(double));
				checkCUDAErrorWithLine("Malloc failed!");
				memset << <blocks, params.block_size >> > (layers[i + 1], dev_z, 0);// zero position because it is a running buffer
				checkCUDAErrorWithLine("Memset failed!");
				vdb.push_back(dev_z);
				// Vw
				blocks = ((layers[i] * layers[i + 1]) + params.block_size - 1) / params.block_size;
				cudaMalloc((void**)&dev_w, (layers[i] * layers[i + 1]) * sizeof(double));
				memset << <blocks, params.block_size >> > ((layers[i] * layers[i + 1]), dev_w, 0); // zero position because it is a running buffer
				checkCUDAErrorWithLine("Cuda malloc failed!");
				vdw.push_back(dev_w);
			}
		}
		// initilizaton cublas handle
		cublasCreate(&handle); 
		// set read_dev_y flag to false (i.e not read data)
		read_dev_y = false;
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
		double *res = new double[params.output_size]();
		assert(n == params.input_size);
		// copy over data to process
		cudaMemcpy(dev_data, data, n * sizeof(double), cudaMemcpyHostToDevice);
		checkCUDAErrorWithLine("Cuda memcpy failed!");
		// reset reduction buffer
		cudaMemset(dev_reduction_pow2 + params.output_size - 1, 0, ((1 << ilog2ceil(params.output_size)) - params.output_size + 1) * sizeof(double));
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
				exp_copy << <blocks, params.block_size >> > (params.layer_sizes[i], a[i], z[i]);
				checkCUDAErrorWithLine("exp copy failed!");
				cudaMemcpy(dev_reduction_pow2, a[i],params.layer_sizes[i]* sizeof(double), cudaMemcpyDeviceToDevice);
				checkCUDAErrorWithLine("dev to dev copy failed!");
				// modified scan to get the exponential sum of all elements (P1 of assignment used)
				reduction(1 << ilog2ceil(params.layer_sizes[i]), dev_reduction_pow2);
				double exp_sum;
				cudaMemcpy(&exp_sum, dev_reduction_pow2 + (1 << ilog2ceil((params.layer_sizes[i]))) - 1, sizeof(double), cudaMemcpyDeviceToHost); // copy last value to cpu
				checkCUDAErrorWithLine("Cuda memcpy failed!");
				// softmax activation
				softmax_activation <<<blocks, params.block_size >>> (params.layer_sizes[i], z[i], a[i], exp_sum);
				checkCUDAErrorWithLine("softmax failed!");
			}
		}
		cudaMemcpy(res, a[params.layer_count - 1], params.layer_sizes[params.layer_count - 1] * sizeof(double), cudaMemcpyDeviceToHost);
		checkCUDAErrorWithLine("Cuda res memcpy failed!");
		// reset reduction buffer (we need to zero only the last part of the array
		cudaMemset(dev_reduction_pow2 + params.output_size - 1, 0, ((1 << ilog2ceil(params.output_size)) - params.output_size + 1) * sizeof(double));
		return res;
	}

	void Net::backprop(double *y) {
		// calculate loss grad
		int n = -1;
		if (!read_dev_y) {
			cudaMemcpy(dev_y, y, params.layer_sizes[params.layer_count - 1] * sizeof(double), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("Cuda memcpy failed!");
			read_dev_y = true;
		}
		for (int i = params.layer_count - 1; i >= 0; i--) {
			n = params.layer_sizes[i];
			blocks = ceil((n + params.block_size - 1) / params.block_size);
			if (i == params.layer_count - 1) { // softmax grad
				bias_addition << <blocks, params.block_size >> > (n, a[i], dev_y, dL_dz[i], -1); // y_hat - y
				checkCUDAErrorWithLine("bias addition failed!");
			}
			else { // relu grad
				relu_grad<< <blocks, params.block_size >> > (n, z[i], d_relu[i]); // relu grad
				checkCUDAErrorWithLine("relu grad failed!");
				element_mult << <blocks, params.block_size >> > (n, d_relu[i], dL_da[i], dL_dz[i]); // dz = da * a'
				checkCUDAErrorWithLine("element wise mult failed!");
			}
			if (i != 0) { // a[i-1] exists + we need to calculate dL_da[i-1]
				gpu_blas_mmul(dL_dz[i], a[i-1], dL_dw[i], params.layer_sizes[i], 1, params.layer_sizes[i-1], false, true); // dw
				checkCUDAErrorWithLine("Matrix mult failed!");
				gpu_blas_mmul(w[i], dL_dz[i], dL_da[i - 1], params.layer_sizes[i - 1], params.layer_sizes[i], 1, true, false); // da
				checkCUDAErrorWithLine("Matrix mult failed!");
				checkCUDAErrorWithLine(("Print error failed! " + to_string(i)).c_str());
			}
			else { // just need tp calculate dL_dw[i]
				gpu_blas_mmul(dL_dz[i], dev_data, dL_dw[i], params.layer_sizes[i], 1, params.input_size, false, true); // dw
				checkCUDAErrorWithLine("Matrix mult failed!");
			}
		}
		// update weights in inverse order
		for (int i = 0; i < params.layer_count; i++) {
			int layer_im1;
			if (i != 0)
				layer_im1 = params.layer_sizes[i - 1];
			else 
				layer_im1 = params.input_size;
			// W
			n = params.layer_sizes[i] * layer_im1;
			blocks = ceil((n + params.block_size - 1) / params.block_size);
			if (momentum_grad) {
				update_momentum <<<blocks, params.block_size >>> (n, vdw[i], dL_dw[i], params.beta);
				checkCUDAErrorWithLine("Update momentum vw failed!");
				update_params << <blocks, params.block_size >> > (n, w[i], vdw[i], params.lr);
				checkCUDAErrorWithLine("Update w failed!");
			}
			else {
				update_params << <blocks, params.block_size >> > (n, w[i], dL_dw[i], params.lr);
				checkCUDAErrorWithLine("Update w failed!");
			}
			//B
			n = params.layer_sizes[i];
			blocks = ceil((n + params.block_size - 1) / params.block_size);
			if (momentum_grad) {
				update_momentum << <blocks, params.block_size >> > (n, vdb[i], dL_dz[i], params.beta);
				checkCUDAErrorWithLine("Update momentum vb failed!");
				update_params << <blocks, params.block_size >> > (n, b[i], vdb[i], params.lr);
				checkCUDAErrorWithLine("Update b failed!");
			}
			else {
				update_params << <blocks, params.block_size >> > (n, b[i], dL_dz[i], params.lr);
				checkCUDAErrorWithLine("Update b failed!");
			}
		}
		read_dev_y = false; // for next cycle
	}
	void Net::reduction(int n, double *dev_odata) {
		// reduce phase
		for (int d = 0; d <= ilog2ceil(n) - 1; d++) {
			// compute number of threads to spawn
			blocks = ceil((n / (1 << (d + 1)) + params.block_size - 1) / params.block_size);
			reduce_kern <<<blocks, params.block_size >> > (n, dev_odata, d);
			checkCUDAErrorWithLine("reduce phase failed!");
		}
	}
	double Net::loss(double *y_pred, double *y) {
		double loss = 0;
		if (!read_dev_y) {
			cudaMemcpy(dev_y, y, params.output_size * sizeof(double), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("Cuda memcpy failed!");
			read_dev_y = true;
		}
		blocks = ceil((params.output_size + params.block_size - 1) / params.block_size);
		cross_entropy_kernal <<< blocks, params.block_size>>> (params.output_size, dev_y, a[params.layer_count - 1], dev_reduction_pow2);
		// reduction to get sum
		reduction(1 << ilog2ceil(params.output_size), dev_reduction_pow2);
		cudaMemcpy(&loss, dev_reduction_pow2 + (1 << ilog2ceil(params.output_size)) - 1, sizeof(double), cudaMemcpyDeviceToHost);
		return -loss;
	}

	void Net::update_lr() {
		params.lr /= 2;
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
		// free momentum variables
		for (auto x : vdw)
			cudaFree(x);
		for (auto x : vdb)
			cudaFree(x);
		cudaFree(dev_data);
		cudaFree(dev_y);
		cudaFree(dev_reduction_pow2);
		// clean culbas hand
		cublasDestroy(handle);
	}
}
