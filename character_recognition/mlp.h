#pragma once

#include "common.h"
#include <vector>
#include <curand.h>
#include <cublas_v2.h>
using namespace std;
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	struct Params {
		vector<int> layer_sizes;
		int layer_count, input_size;
		double lr;
		const static int block_size = 128;
	};
	class Net {
		// input data
		double *dev_data;
		// cublas
		cublasHandle_t handle;
		vector<double *> w;
		vector<double *> b;
		// intermediate results
		vector<double *> g;
		vector<double *> a;
		int blocks;
		// gradient variables
		vector<double *> dL_dg;
		vector<double *> da_dg; // activation function grads (dy_dg for the final layer)
		double * dL_dyhat;
		// parameters
		Params params;
		// helperfunctions for the class
		// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
		void GPU_fill_rand(double *A, int size, double std);
		// multiplication of matrices
		void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n);
	public:
		Net(int n, vector<int> layers, double lr);	// creates weight matrixs
		double* forward(double *data, int n); // returns class
		void backprop(double *y); 
		double Net::loss(int *y);
		~Net(); // to delete the weights
	};
    // TODO: implement required elements for MLP sections 1 and 2 here
}
// From P1 of this assignment, shared memory scan (modified to use exponents of values
