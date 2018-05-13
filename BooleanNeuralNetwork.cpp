#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "include/OneLayerNeuralNetwork.h"

using namespace std;

int main() {
	const size_t inputsSize = 3; // два входа и один для смещения
	const size_t hiddensSize = 4; // два скрытых нейрона
	const size_t outputsSize = 3; // три выхода

	const double alpha = 0.1; // скорость обучения
	const double eps = 1e-3; // точность обучения
	const size_t maxEpoch = 1000000; // максимальное число эпох

	OneLayerNeuralNetwork network(inputsSize, hiddensSize, outputsSize);

	vector<vector<double>> learnInputData = { 
		vector<double> { 0, 0, 1 },
		vector<double> { 0, 1, 1 }, 
		vector<double> { 1, 0, 1 }, 
		vector<double> { 1, 1, 1 }
	};

	vector<vector<double>> learnOutputData = { 
		vector<double> { 0, 0, 0 }, 
		vector<double> { 1, 1, 0 }, 
		vector<double> { 1, 1, 0 }, 
		vector<double> { 0, 1, 1 },
	};

	vector<string> names = {
		"xor", "or", "and"
	};

	network.Train(learnInputData, learnOutputData, alpha, eps, maxEpoch, false);

	network.PrintState();

	for (size_t p = 0; p < learnInputData.size(); p++) {
		vector<double> results = network.GetResult(learnInputData[p]);

		for (size_t i = 0; i < results.size(); i++) {
			cout << learnInputData[p][0] << " " << names[i] << " " << learnInputData[p][1] << " = " << results[i] << "\t";
		}

		cout << endl;
	}
}