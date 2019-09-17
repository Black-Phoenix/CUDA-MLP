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
		double *dev_y;
		// cublas
		cublasHandle_t handle;
		vector<double *> w;
		vector<double *> b;
		// intermediate results
		vector<double *> z; // z = w * x + b
		vector<double *> a; // 
		int blocks;
		// gradient variables
		vector<double *> dL_dz; // z = wx + b
		vector<double *> dL_da; // activation function grads (dy_dg for the final layer)
		vector<double *> dL_dw; 
		// relu gradient
		vector<double *> d_relu;
		// parameters
		Params params;
		// helperfunctions for the class
		// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
		void GPU_fill_rand(double *A, int size, double std);
		// multiplication of matrices
		void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, bool trans_flag_a = false, bool trans_flag_b = false);
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
