#pragma once

#include <iostream>
#include <vector>
#include "Neuron.h"

class OneLayerNeuralNetwork {
	size_t inputsSize;
	size_t hiddensSize;
	size_t outputsSize;
	size_t trainEpoch;

	std::vector<Neuron> inputs;
	std::vector<Neuron> hiddens;
	std::vector<Neuron> outputs;

	void ReshapeWeights();

public:
	OneLayerNeuralNetwork(size_t inputsSize, size_t hiddensSize, size_t outputsSize);
	OneLayerNeuralNetwork(size_t inputsSize, size_t hiddensSize, size_t outputsSize, ActivationPointer activation, ActivationPointer derivative);

	std::vector<double> GetResult(const std::vector<double>& inputData);
	void PrintState() const;
	void Train(const std::vector<std::vector<double>>& learnInputData, const std::vector<std::vector<double>>& learnOutputData, double alpha = 0.1, double eps = 1e-5, size_t maxEpoch = 100000, bool reshape = true);
};