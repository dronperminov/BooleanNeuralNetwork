#include "OneLayerNeuralNetwork.h"

using namespace std;

OneLayerNeuralNetwork::OneLayerNeuralNetwork(size_t inputsSize, size_t hiddensSize, size_t outputsSize) {
	this->inputsSize = inputsSize;
	this->hiddensSize = hiddensSize;
	this->outputsSize = outputsSize;

	inputs = vector<Neuron>(inputsSize, Neuron(NeuronType::input));
	hiddens = vector<Neuron>(hiddensSize, Neuron(NeuronType::hidden, inputsSize));
	outputs = vector<Neuron>(outputsSize, Neuron(NeuronType::output, hiddensSize));

	trainEpoch = 0;
}

OneLayerNeuralNetwork::OneLayerNeuralNetwork(size_t inputsSize, size_t hiddensSize, size_t outputsSize, ActivationPointer activation, ActivationPointer derivative) {
	this->inputsSize = inputsSize;
	this->hiddensSize = hiddensSize;
	this->outputsSize = outputsSize;

	inputs = vector<Neuron>(inputsSize, Neuron(NeuronType::input));
	hiddens = vector<Neuron>(hiddensSize, Neuron(NeuronType::hidden, inputsSize, activation, derivative));
	outputs = vector<Neuron>(outputsSize, Neuron(NeuronType::output, hiddensSize));

	trainEpoch = 0;
}

// получение выходов нейросети
vector<double> OneLayerNeuralNetwork::GetResult(const vector<double>& inputData) {
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
void OneLayerNeuralNetwork::PrintState() const {
	cout << endl << "State: " << endl;
	for (size_t i = 0; i < inputsSize; i++)
		inputs[i].Print();

	cout << endl;
	for (size_t i = 0; i < hiddensSize; i++)
		hiddens[i].Print();

	cout << endl;
	for (size_t i = 0; i < outputsSize; i++)
		outputs[i].Print();

	cout << endl << "Train epoch: " << trainEpoch << endl;
}

// улучшение обучения
void OneLayerNeuralNetwork::ReshapeWeights() {
	double beta = 0.7 * pow(hiddensSize, 1.0 / inputsSize);

	for (size_t i = 0; i < hiddensSize; i++) {
		double w = 0;

		for (size_t j = 0; j < inputsSize; j++)
			w += hiddens[i].GetWeight(j) * hiddens[i].GetWeight(j);

		for (size_t j = 0; j < inputsSize; j++)
			hiddens[i].SetWeight(j, beta * hiddens[i].GetWeight(j) / sqrt(w)); 
	}
}

// обучение сети
void OneLayerNeuralNetwork::Train(const vector<vector<double>>& learnInputData, const vector<vector<double>>& learnOutputData, double alpha, double eps, size_t maxEpoch, bool reshape) {
	if (reshape)
		ReshapeWeights();
	
	trainEpoch = 0;
	double gError;

	do {
		gError = 0;
		trainEpoch++;

		for (size_t p = 0; p < learnInputData.size(); p++) {
			vector<double> results = GetResult(learnInputData[p]);

			vector<double> sigmas = vector<double>(outputsSize);
			for (size_t i = 0; i < outputsSize; i++) {
				sigmas[i] = learnOutputData[p][i] - results[i];
				gError += sigmas[i] * sigmas[i];
			}

			vector<double> errors = vector<double>(hiddensSize, 0);
			for (size_t i = 0; i < outputsSize; i++) {
				for (size_t j = 0; j < hiddensSize; j++) {
					errors[j] += sigmas[i] * outputs[i].GetWeight(j);
				}
			}

			for (size_t i = 0; i < hiddensSize; i++) {
				for (size_t j = 0; j < inputsSize; j++) {
					double weight = hiddens[i].GetWeight(j);
					hiddens[i].SetWeight(j, weight + alpha * errors[i] * inputs[j].GetOutput() * hiddens[i].GetDerivativeOutput());
				}
			}

			for (size_t i = 0; i < outputsSize; i++) {
				for (size_t j = 0; j < hiddensSize; j++) {
					double weight = outputs[i].GetWeight(j);
					outputs[i].SetWeight(j, weight + alpha * sigmas[i] * hiddens[j].GetOutput());
				}
			}
		}
	} while (sqrt(gError) > eps && trainEpoch < maxEpoch);

	if (trainEpoch == maxEpoch)
		cout << "Warning! Max epoch reached" << endl;
}