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
//#include <opencv2/core/core.hpp>
#include <windows.h>
//#include <opencv2/highgui/highgui.hpp>
#define inputs 196
#define classes 52
#define lr 0.2
using namespace std;
//int image_read(string path) {
//	cv::Mat image = cv::imread(path.c_str(), 0);
//	if (!image.data)                              // Check for invalid input
//	{
//		cout << "Could not open or find the image" << std::endl;
//		return -1;
//	}
//	// 1d array!!!
//	uchar * arr = image.isContinuous() ? image.data : image.clone().data;
//	uint length = image.total()*image.channels();
//	return length;
//}

void read_directory(const std::string& name, vector<string>& v)
{
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			// todo take only file names with bmp/img
			v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}
int main(int argc, char* argv[]) {
	// test read dir
	/*string path = R"(..\data-set\)";
	vector<string> files;
	read_directory(path, files);
	for (auto x : files)
		cout << x << endl;*/
	// test forward pass
	CharacterRecognition::Net nn(inputs, {98, 52, 52, classes}, lr);
	double data[inputs] = {0};
	auto x = nn.forward(data, inputs);
//	nn.backprop(data);
	double sum = 0;
	for (int i = 0; i < classes; i++) {
		cout << x[i] << ", " << endl;
		sum += x[i];
	}
	cout << sum << endl;
	delete[] x;
	return 0;
}
