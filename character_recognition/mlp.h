#pragma once

#include "common.h"
#include <vector>
#include <curand.h>
#include <cublas_v2.h>
using namespace std;
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
//namespace Timer {
//	class PerformanceTimer
//	{
//	public:
//		PerformanceTimer()
//		{
//			cudaEventCreate(&event_start);
//			cudaEventCreate(&event_end);
//		}
//
//		~PerformanceTimer()
//		{
//			cudaEventDestroy(event_start);
//			cudaEventDestroy(event_end);
//		}
//
//		void startCpuTimer()
//		{
//			if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
//			cpu_timer_started = true;
//
//			time_start_cpu = std::chrono::high_resolution_clock::now();
//		}
//
//		void endCpuTimer()
//		{
//			time_end_cpu = std::chrono::high_resolution_clock::now();
//
//			if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }
//
//			std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
//			prev_elapsed_time_cpu_milliseconds =
//				static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());
//
//			cpu_timer_started = false;
//		}
//
//		void startGpuTimer()
//		{
//			if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
//			gpu_timer_started = true;
//
//			cudaEventRecord(event_start);
//		}
//
//		void endGpuTimer()
//		{
//			cudaEventRecord(event_end);
//			cudaEventSynchronize(event_end);
//
//			if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }
//
//			cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
//			gpu_timer_started = false;
//		}
//
//		float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
//		{
//			return prev_elapsed_time_cpu_milliseconds;
//		}
//
//		float getGpuElapsedTimeForPreviousOperation() //noexcept
//		{
//			return prev_elapsed_time_gpu_milliseconds;
//		}
//
//		// remove copy and move functions
//		PerformanceTimer(const PerformanceTimer&) = delete;
//		PerformanceTimer(PerformanceTimer&&) = delete;
//		PerformanceTimer& operator=(const PerformanceTimer&) = delete;
//		PerformanceTimer& operator=(PerformanceTimer&&) = delete;
//
//	private:
//		cudaEvent_t event_start = nullptr;
//		cudaEvent_t event_end = nullptr;
//
//		using time_point_t = std::chrono::high_resolution_clock::time_point;
//		time_point_t time_start_cpu;
//		time_point_t time_end_cpu;
//
//		bool cpu_timer_started = false;
//		bool gpu_timer_started = false;
//
//		float prev_elapsed_time_cpu_milliseconds = 0.f;
//		float prev_elapsed_time_gpu_milliseconds = 0.f;
//	};
//}
namespace CharacterRecognition {
	Common::PerformanceTimer& timer();
	struct Params {
		vector<int> layer_sizes;
		int layer_count, input_size;
		int output_size;
		double lr, beta;
		const static int block_size = 128;
	};
	class Net {
		bool momentum_grad; // just a flag for momentum
		// input data
		double *dev_data;
		double *dev_y;
		double *dev_reduction_pow2;
		bool read_dev_y;
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
		// momentum gradients
		vector<double *> vdb;
		vector<double *> vdw;
		// relu gradient
		vector<double *> d_relu;
		// parameters
		Params params;
		// helperfunctions for the class
		// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
		void GPU_fill_rand(double *A, int size, double std);
		// multiplication of matrices
		void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n, bool trans_flag_a = false, bool trans_flag_b = false);
		// reduction invocation code
		void Net::reduction(int n, double *dev_odata);
	public:
		Net(int n, vector<int> layers, double lr, double beta = -1);	// creates weight matrixs
		double* forward(double *data, int n); // returns class
		void backprop(double *y); 
		double Net::loss(double *y_pred, double *y);
		void update_lr(); // halfs learning rate every x iterations
		void dump_weights(string path); 
		~Net(); // to delete the weights
	};
}
