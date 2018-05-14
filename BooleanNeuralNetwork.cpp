#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include "include/OneLayerNeuralNetwork.h"

using namespace std;

int main() {
	const size_t inputsSize = 2; // два входа
	const size_t hiddensSize = 4; // два скрытых нейрона
	const size_t outputsSize = 4; // три выхода

	const double alpha = 0.1; // скорость обучения
	const double eps = 1e-5; // точность обучения
	const size_t maxEpoch = 1000000; // максимальное число эпох

	OneLayerNeuralNetwork network(inputsSize, hiddensSize, outputsSize);

	vector<vector<double>> learnInputData = { 
		vector<double> { 0, 0 },
		vector<double> { 0, 1 }, 
		vector<double> { 1, 0 }, 
		vector<double> { 1, 1 }
	};

	vector<vector<double>> learnOutputData = { 
		vector<double> { 0, 0, 0, 1 }, 
		vector<double> { 1, 1, 0, 1 }, 
		vector<double> { 1, 1, 0, 0 }, 
		vector<double> { 0, 1, 1, 1 },
	};

	vector<string> names = {
		"^", "|", "&", "->"
	};

	network.Train(learnInputData, learnOutputData, alpha, eps, maxEpoch, false);

	network.PrintState();

	cout <<  fixed;

	for (size_t p = 0; p < learnInputData.size(); p++) {
		vector<double> results = network.GetResult(learnInputData[p]);

		for (size_t i = 0; i < results.size(); i++) {
			cout << setprecision(0) << learnInputData[p][0] << " " << names[i] << " " << learnInputData[p][1] << " = " << left << setw(15) << setprecision(5) << results[i];
		}

		cout << endl;
	}
}