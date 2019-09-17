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

#define classes 52
#define epochs 40000
#define in_dim 15
#define inputs in_dim*in_dim
#define lr 0.0038
#define beta 0.65
using namespace std;

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
			if (string(data.cFileName).find("bmp") != std::string::npos)
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
int main(int argc, char* argv[]) {
	// load image paths
	string path = R"(..\data-set\)";
	vector<string> files;
	vector<double *> input_data;
	vector<double *> output_data;
	read_directory(path, files);
	for (auto x : files)
		if (image_read(path + x, input_data))
			lable_read(x, output_data);
	// test forward pass
	
	CharacterRecognition::Net nn(inputs, {98, 65, 50, 30, 25, 40, classes}, lr, beta);
	int i;
	float total_loss = 0;
	for (i = 0; i < epochs; i++) {
		auto x = nn.forward(input_data[i%classes], inputs);
		//csv_write(i, nn.loss(x, output_data[i%classes]), "loss.csv");
		total_loss += nn.loss(x, output_data[i%classes]);
		if (i % classes == 0 && i) {
			cout << i << ":" << total_loss/classes << endl;
			total_loss = 0;
		}
		nn.backprop(output_data[i%classes]);
		if (!(i % (classes * 100)) && i) 
			nn.update_lr();
		delete[] x;
	}
	int val = 0;
	for (int i = 0; i < classes; i++) {
		double* x = nn.forward(input_data[i], inputs);
		int pos1 = distance(x, max_element(x, x + classes));
		int pos2 = distance(output_data[i], max_element(output_data[i], output_data[i] + classes));
		val += (bool)(pos1 == pos2);
		delete[] x;
	}
	cout << "Passes " << val << " Out of "<<classes;
	// Free up data
	for (auto x : input_data)
		delete[] x;
	for (auto x : output_data)
		delete[] x;
	return 0;
}
