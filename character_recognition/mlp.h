#pragma once

#include "common.h"
#include <vector>
#include <curand.h>
#include <cublas_v2.h>
using namespace std;
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	class Net {
		// cublas
		cublasHandle_t handle;
		vector<float *> w;
		vector<float *> b;
		// intermediate results
		vector<float *> g;
		vector<float *> a;
		const static int block_size = 128;
		int blocks;
		int input_size, layer_count;
		vector<int> layer_sizes;
		// gradient variables
		vector<float *> dL_dw; // g = x*w + b
		vector<float *> dL_db; // g = x*w + b
		vector<float *> da_dg; // activation function grads (dy_dg for the final layer)
		// helperfunctions for the class

		// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
		void GPU_fill_rand(float *A, int size, float std);
		// multiplication of matrices
		void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n);
	public:
		Net(int n, vector<int> layers);	// creates weight matrixs
		float* forward(float *data, int n); // returns class
		void backprop(int *output); // returns loss
		~Net(); // to delete the weights
	};
    // TODO: implement required elements for MLP sections 1 and 2 here
}
// From P1 of this assignment, shared memory scan (modified to use exponents of values
