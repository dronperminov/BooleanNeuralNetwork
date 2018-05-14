#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "include/OneLayerNeuralNetwork.h"

using namespace std;

double f(double x) {
	return sin(x);
}

int main() {
	const size_t inputsSize = 2; // один вход
	const size_t hiddensSize = 6; // шесть скрытых нейронов
	const size_t outputsSize = 1; // один выход

	const double alpha = 0.001;
	const double eps = 1e-2;
	const size_t maxEpoch = 100000;

	OneLayerNeuralNetwork network(inputsSize, hiddensSize, outputsSize);

	double a = -3;
	double b = 3;
	size_t n = 30;

	vector<vector<double>> learnInputData;
	vector<vector<double>> learnOutputData;

	for (size_t i = 0; i < n; i++) {
		double x = a + (b - a) * ((double) rand() / RAND_MAX); // [a, b]

		learnInputData.push_back({ x, 1 });
		learnOutputData.push_back({ f(x) });
	}

	network.Train(learnInputData, learnOutputData, alpha, eps, maxEpoch);

	network.PrintState();

	for (size_t p = 0; p < learnInputData.size(); p++) {
		vector<double> results = network.GetResult(learnInputData[p]);
		double result = results[0];
		double correct = learnOutputData[p][0];
		double error = fabs(result - correct);

		cout << "f(" << learnInputData[p][0] << ") = " << results[0] << ", must: " << correct << ", error: " << error << endl;
	}
}