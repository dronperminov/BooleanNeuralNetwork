#include <iostream>
#include <vector>
#include <cmath>

#include "include/Neuron.h"

using namespace std;

// получение выходов нейросети
vector<double> GetResult(vector<Neuron>& inputs, vector<Neuron>& hiddens, vector<Neuron> &outputs, size_t inputsSize, size_t hiddensSize, size_t outputsSize, vector<double>& inputData) {
	for (size_t i = 0; i < inputsSize; i++) {
		inputs[i].SetInput(0, inputData[i]);
	}

	for (size_t i = 0; i < inputsSize; i++) {
		for (size_t j = 0; j < hiddensSize; j++) {
			hiddens[j].SetInput(i, inputs[i].GetOutput());
		}
	}

	for (size_t i = 0; i < outputsSize; i++) {
		for (size_t j = 0; j < hiddensSize; j++) {
			outputs[i].SetInput(j, hiddens[j].GetOutput());
		}
	}

	vector<double> results = vector<double>(outputsSize);

	for (size_t i = 0; i < outputsSize; i++)
		results[i] = outputs[i].GetOutput();

	return results;   
}

// печать состояний сети (всех входных, скрытых и выходных нейронов)
void PrintState(vector<Neuron>& inputs, vector<Neuron> hiddens, vector<Neuron> &outputs, size_t inputsSize, size_t hiddensSize, size_t outputsSize) {
	cout << endl << "State: " << endl;
	for (size_t i = 0; i < inputsSize; i++)
		inputs[i].Print();

	cout << endl;
	for (size_t i = 0; i < hiddensSize; i++)
		hiddens[i].Print();

	cout << endl;
	for (size_t i = 0; i < outputsSize; i++)
		outputs[i].Print();

	cout << endl;
}

double step(double x) {
	return x > 0.5 ? 1 : 0;
}

double stepD(double x) {
	return 1;
}

int main() {
	size_t inputsSize = 2; // два входа
	size_t hiddensSize = 2; // два скрытых нейрона
	size_t outputsSize = 3; // три выхода

	vector<Neuron> inputs(inputsSize, Neuron(NeuronType::input));
	vector<Neuron> hiddens(hiddensSize, Neuron(NeuronType::hidden, inputsSize, step, stepD));
	vector<Neuron> outputs(outputsSize, Neuron(NeuronType::output, hiddensSize, step, stepD));

	vector<vector<double>> learnInputData = { 
		vector<double> { 0, 0 },
		vector<double> { 0, 1 }, 
		vector<double> { 1, 0 }, 
		vector<double> { 1, 1 }
	};

	vector<vector<double>> learnOutputData = { 
		vector<double> { 0, 0, 0 }, 
		vector<double> { 1, 1, 0 }, 
		vector<double> { 1, 1, 0 }, 
		vector<double> { 0, 1, 1 }
	};

	double alpha = 0.01;
	double beta = 0.7 * pow(hiddensSize, 1.0 / inputsSize);

	double gError;

	for (size_t i = 0; i < hiddensSize; i++) {
		double w = 0;
		for (size_t j = 0; j < inputsSize; j++)
			w += hiddens[i].GetWeight(j) * hiddens[i].GetWeight(j);

		for (size_t j = 0; j < inputsSize; j++)
			hiddens[i].SetWeight(j, beta * hiddens[i].GetWeight(j) / sqrt(w)); 
	}

	do {
		gError = 0;

		for (size_t p = 0; p < learnInputData.size(); p++) {
			vector<double> results = GetResult(inputs, hiddens, outputs, inputsSize, hiddensSize, outputsSize, learnInputData[p]);

			vector<double> sigmas = vector<double>(outputsSize);
			for (size_t i = 0; i < outputsSize; i++) {
				sigmas[i] = learnOutputData[p][i] - results[i];
				gError += fabs(sigmas[i]);
			}

			vector<double> errors = vector<double>(hiddensSize);
			for (size_t i = 0; i < outputsSize; i++) {
				for (size_t j = 0; j < hiddensSize; j++) {
					errors[j] += sigmas[i] * outputs[i].GetWeight(j);
				}
			}

			for (size_t i = 0; i < hiddensSize; i++) {
				for (size_t j = 0; j < inputsSize; j++) {
					double weight = hiddens[i].GetWeight(j);
					hiddens[i].SetWeight(j, weight + alpha * errors[i] * inputs[j].GetOutput());
				}
			}

			for (size_t i = 0; i < outputsSize; i++) {
				for (size_t j = 0; j < hiddensSize; j++) {
					double weight = outputs[i].GetWeight(j);
					outputs[i].SetWeight(j, weight + alpha * sigmas[i] * hiddens[j].GetOutput());
				}
			}
		}
	} while (gError > 0);

	PrintState(inputs, hiddens, outputs, inputsSize, hiddensSize, outputsSize);

	for (size_t p = 0; p < learnInputData.size(); p++) {
		vector<double> results = GetResult(inputs, hiddens, outputs, inputsSize, hiddensSize, outputsSize, learnInputData[p]);

		cout << learnInputData[p][0] << " ^ " << learnInputData[p][1] << " = " << results[0];
		cout << "\t" << learnInputData[p][0] << " or " << learnInputData[p][1] << " = " << results[1];
		cout << "\t" << learnInputData[p][0] << " and " << learnInputData[p][1] << " = " << results[2] << endl;
	}
}