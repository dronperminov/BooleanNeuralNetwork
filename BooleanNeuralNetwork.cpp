#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "include/Neuron.h"

using namespace std;

double step(double x) {
	return x > 0.5 ? 1 : 0;
}

double stepD(double x) {
	return 1;
}

class NeuralNetwork {
	size_t inputsSize;
	size_t hiddensSize;
	size_t outputsSize;

	vector<Neuron> inputs;
	vector<Neuron> hiddens;
	vector<Neuron> outputs;

	void ReshapeWeights(double beta);

public:
	NeuralNetwork(size_t inputsSize, size_t hiddensSize, size_t outputsSize);

	vector<double> GetResult(const vector<double>& inputData);
	void PrintState() const;
	void Train(double alpha, double beta, const vector<vector<double>>& learnInputData, const vector<vector<double>>& learnOutputData);
};

NeuralNetwork::NeuralNetwork(size_t inputsSize, size_t hiddensSize, size_t outputsSize) {
	this->inputsSize = inputsSize;
	this->hiddensSize = hiddensSize;
	this->outputsSize = outputsSize;

	inputs = vector<Neuron>(inputsSize, Neuron(NeuronType::input));
	hiddens = vector<Neuron>(hiddensSize, Neuron(NeuronType::hidden, inputsSize, step, stepD));
	outputs = vector<Neuron>(outputsSize, Neuron(NeuronType::output, hiddensSize, step, stepD));
}

// получение выходов нейросети
vector<double> NeuralNetwork::GetResult(const vector<double>& inputData) {
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
void NeuralNetwork::PrintState() const {
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

// улучшение обучения
void NeuralNetwork::ReshapeWeights(double beta) {
	for (size_t i = 0; i < hiddensSize; i++) {
		double w = 0;

		for (size_t j = 0; j < inputsSize; j++)
			w += hiddens[i].GetWeight(j) * hiddens[i].GetWeight(j);

		for (size_t j = 0; j < inputsSize; j++)
			hiddens[i].SetWeight(j, beta * hiddens[i].GetWeight(j) / sqrt(w)); 
	}
}

// обучение сети
void NeuralNetwork::Train(double alpha, double beta, const vector<vector<double>>& learnInputData, const vector<vector<double>>& learnOutputData) {
	ReshapeWeights(beta);

	double gError;

	do {
		gError = 0;

		for (size_t p = 0; p < learnInputData.size(); p++) {
			vector<double> results = GetResult(learnInputData[p]);

			vector<double> sigmas = vector<double>(outputsSize);
			for (size_t i = 0; i < outputsSize; i++) {
				sigmas[i] = learnOutputData[p][i] - results[i];
				gError += sigmas[i] * sigmas[i];
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
	} while (sqrt(gError) > 0);
}

int main() {
	const size_t inputsSize = 2; // два входа
	const size_t hiddensSize = 2; // два скрытых нейрона
	const size_t outputsSize = 3; // три выхода

	NeuralNetwork network(inputsSize, hiddensSize, outputsSize);	

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
		vector<double> { 0, 1, 1 },
	};

	vector<string> names = {
		"xor", "or", "and"
	};

	double alpha = 0.01;
	double beta = 0.7 * pow(hiddensSize, 1.0 / inputsSize);

	network.Train(alpha, beta, learnInputData, learnOutputData);

	network.PrintState();

	for (size_t p = 0; p < learnInputData.size(); p++) {
		vector<double> results = network.GetResult(learnInputData[p]);

		for (size_t i = 0; i < results.size(); i++) {
			cout << learnInputData[p][0] << " " << names[i] << " " << learnInputData[p][1] << " = " << results[i] << "\t";
		}

		cout << endl;
	}
}