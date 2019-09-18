/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Vaibhav Arcot
 * @date      2019
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"
#include <cublas_v2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <windows.h>
#include <fstream>
#include <random>

#define classes 26
#define epochs 4000
#define in_dim 15
#define inputs in_dim*in_dim
#define lr 0.004
#define beta 0.35
#define char_74k false

using namespace std;

float avg_time_forward = 0, avg_time_backward = 0;

int image_read(string path, vector<double *> &data) {
	cv::Mat image = cv::imread(path.c_str(), 0);
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return 0;
	}
	data.push_back(new double[inputs]);
	cv::resize(image, image, cv::Size(in_dim, in_dim));
	for (int i = 0; i < inputs; i++)
		data[data.size() - 1][i] = (double)image.data[i];
	return 1;
}

double * image_read_rot(string path, int rot) {
	cv::Mat image = cv::imread(path.c_str(), 0);
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		cout << "failed on " << path << endl;
		return 0;
	}
	double *data = new double[inputs];
	cv::Point2f pc(image.cols / 2., image.rows / 2.);
	cv::Mat r = cv::getRotationMatrix2D(pc, rot, 1.0);

	cv::warpAffine(image, image, r, image.size()); // what size I should use?
	cv::resize(image, image, cv::Size(in_dim, in_dim));
	for (int i = 0; i < inputs; i++)
		data[i] = (double)image.data[i];
	return data;
}
void csv_write(int iter, double loss, string file_name) {
	std::ofstream outfile;
	outfile.open(file_name, std::ios_base::app);
	outfile << iter << "," << loss << endl;
}

void read_directory(const std::string& name, vector<string>& v) {
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			if (string(data.cFileName).find("bmp") != std::string::npos || string(data.cFileName).find("png") != std::string::npos)
				v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

void lable_read(string name, vector<double *> &data) {
	int value = stoi(name.substr(0, 2));
	data.push_back(new double[classes]);
	memset(data[data.size() - 1], 0, classes * sizeof(double));
	data[data.size() - 1][value - 1] = 1;
}
double * lable_read(string name) {
	int value = stoi(name.substr(0, 2));
	double *data = new double[classes];
	memset(data, 0, classes * sizeof(double));
	data[value - 1] = 1;
	return data;
}
int random_int(int min, int max) {
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> uint_dist(min, max); // distribution in range [1, 6]
	return uint_dist(rng);
}

void sequential_training(CharacterRecognition::Net &nn, vector<double *> &input_data,
	vector<double *> &output_data, float &total_loss, int i) {
	auto x = nn.forward(input_data[i%classes], inputs);
	avg_time_forward += CharacterRecognition::timer().getGpuElapsedTimeForPreviousOperation();
	total_loss += nn.loss(x, output_data[i%classes]);
	if (i % classes == 0 && i) {
		cout << i << ":" << total_loss / classes << endl;
		csv_write(i, total_loss / classes, "loss_momentum.csv");
		total_loss = 0;
	}
	if (!(i % (classes * 100)) && i)
		nn.update_lr();
	nn.backprop(output_data[i%classes]);
	avg_time_backward += CharacterRecognition::timer().getGpuElapsedTimeForPreviousOperation();
	delete[] x;
}

void random_training(CharacterRecognition::Net &nn, vector<double *> &input_data,
	vector<double *> &output_data, float &total_loss, int i) {
	int n = random_int(0, input_data.size() - 1);
	auto x = nn.forward(input_data[n], inputs);
	avg_time_forward += CharacterRecognition::timer().getGpuElapsedTimeForPreviousOperation();
	total_loss += nn.loss(x, output_data[n]);
	if (i % classes == 0 && i) {
		cout << i << ":" << total_loss / classes << endl;
		csv_write(i, total_loss / classes, "loss_momentum.csv");
		total_loss = 0;
	}
	/*if (!(i % (classes * 100)) && i)
		nn.update_lr();*/
	nn.backprop(output_data[n]);
	avg_time_backward += CharacterRecognition::timer().getGpuElapsedTimeForPreviousOperation();
	delete[] x;
}

void random_training_rot(string path, CharacterRecognition::Net &nn, const vector<string> &files, float &total_loss, int n) {
	int i = random_int(0, classes - 1);
	int rot = random_int(-10, 10);// +- 10 degree rotation
	string name = path + string(files[i]);
	auto input = image_read_rot(name, rot);
	auto output = lable_read(files[i]);
	auto x = nn.forward(input, inputs);
	avg_time_forward += CharacterRecognition::timer().getGpuElapsedTimeForPreviousOperation();
	total_loss += nn.loss(x, output);
	if (n % classes == 0 && n) {
		cout << n << ":" << total_loss / classes << endl;
		csv_write(n, total_loss / classes, "Rot_loss.csv");
		total_loss = 0;
	}
	if (!(n % (classes * 100)) && n)
		nn.update_lr();
	nn.backprop(output);
	avg_time_backward += CharacterRecognition::timer().getGpuElapsedTimeForPreviousOperation();
	delete[] input;
	delete[] output;
	delete[] x;
}

void lable_read74k(string name, vector<double *> &data) {
	int value = stoi(name.substr(3, 3));
	data.push_back(new double[classes]);
	memset(data[data.size() - 1], 0, classes * sizeof(double));
	data[data.size() - 1][value - 11] = 1;
}

int main(int argc, char* argv[]) {
	// load image paths
	string path = R"(..\data-set\)";// change to ..\data-set\ for regular dataset
	vector<string> files;
	vector<double *> input_data;
	vector<double *> output_data;
	read_directory(path, files);
	for (auto x : files)
		if (image_read(path + x, input_data)) {
			cout << x << endl;
			if(!char_74k)
				lable_read(x, output_data);
			else {
				lable_read74k(x, output_data);
			}

		}
	// forward pass
	CharacterRecognition::Net nn(inputs, {98, 60, 50, 40, 30, classes}, lr, beta);
	int i;
	float total_loss = 0;
	for (i = 0; i < epochs; i++) {
			if(char_74k)
				random_training(nn, input_data, output_data, total_loss, i);
			else {
				//random_training_rot(path, nn, files, total_loss, i);
				sequential_training(nn, input_data, output_data, total_loss, i);
			}
	}
	cout << " Avg forward = " << avg_time_forward / epochs << ", Avg backward = " << avg_time_backward / epochs << endl;
	int val = 0;
	if (!char_74k) {
		for (int i = 0; i < classes; i++) {
			double* x = nn.forward(input_data[i], inputs);
			int pos1 = distance(x, max_element(x, x + classes));
			int pos2 = distance(output_data[i], max_element(output_data[i], output_data[i] + classes));
			val += (bool)(pos1 == pos2);
			delete[] x;
		}
	}
	else {
		// read validation data
		string path = R"(..\74k_dataset_test\)";// change to ..\data-set\ for regular dataset
		vector<string> files;
		vector<double *> input_data_test;
		vector<double *> output_data_test;
		read_directory(path, files);
		for (auto x : files)
			if (image_read(path + x, input_data_test)) {
				cout << x << endl;
				lable_read74k(x, output_data_test);

			}
		for (int i = 0; i < input_data_test.size(); i++) {
			double* x = nn.forward(input_data_test[i], inputs);
			int pos1 = distance(x, max_element(x, x + classes));
			int pos2 = distance(output_data_test[i], max_element(output_data_test[i], output_data_test[i] + classes));
			val += (bool)(pos1 == pos2);
			delete[] x;
		}
		for (auto x : input_data_test)
			delete[] x;
		for (auto x : output_data_test)
			delete[] x;
	}
	cout << "Passes " << val << " Out of "<<classes;
	nn.dump_weights("weights.csv");
	// Free up data
	for (auto x : input_data)
		delete[] x;
	for (auto x : output_data)
		delete[] x;
	return 0;
}
