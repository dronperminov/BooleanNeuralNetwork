#include "Neuron.h"

using namespace std;

double Sigmoid(double x) {
	return 1.0 / (1 + exp(-x));
}

double SigmoidDerivative(double x) {
	return Sigmoid(x) * (1 - Sigmoid(x));
}

double HiperbolicTangent(double x) {
	return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

double HiperbolicTangentDerivative(double x) {
	return 4 * exp(2 * x) / pow(exp(2 * x) + 1, 2);
}

double ReLU(double x) {
	return x > 0 ? x : 0;
}

double ReLUDerivative(double x) {
	return x > 0 ? 1 : 0;
}

Neuron::Neuron(NeuronType type, size_t inputsSize) {
	this->inputsSize = inputsSize;
	this->type = type;

	inputs = vector<double>(inputsSize);
	weights = vector<double>(inputsSize);

	if (type != NeuronType::input) {
		srand(time(NULL));

		for (size_t i = 0; i < inputsSize; i++)
			weights[i] = -0.5 + (double) rand() / RAND_MAX;
	}
	else {
		for (size_t i = 0; i < inputsSize; i++)
			weights[i] = 1;
	}

	ActivationFunction = Sigmoid;
	ActivationDerivativeFunction = SigmoidDerivative;
}

Neuron::Neuron(NeuronType type, size_t inputsSize, ActivationPointer ActivationFunction, ActivationPointer ActivationDerivativeFunction) {
	this->inputsSize = inputsSize;
	this->type = type;

	inputs = vector<double>(inputsSize);
	weights = vector<double>(inputsSize);

	if (type != NeuronType::input) {
		srand(time(NULL));

		for (size_t i = 0; i < inputsSize; i++)
			weights[i] = (double) rand() / RAND_MAX;
	}
	else {
		for (size_t i = 0; i < inputsSize; i++)
			weights[i] = 1;
	}

	this->ActivationFunction = ActivationFunction;
	this->ActivationDerivativeFunction = ActivationDerivativeFunction;
}

double Neuron::GetInput(size_t index) const {
	return inputs[index];
}

std::vector<double> Neuron::GetInputs() const {
	return inputs;
}

void Neuron::SetInputs(const std::vector<double>& inputs) {
	this->inputs = inputs;
}

std::vector<double> Neuron::GetWeights() const {
	return weights;
}

void Neuron::SetWeights(const std::vector<double>& weights) {
	this->weights = weights;
}

void Neuron::SetInput(size_t index, double input) {
	inputs[index] = input;
}

double Neuron::GetWeight(size_t index) const {
	return weights[index];
}

void Neuron::SetWeight(size_t index, double weight) {
	weights[index] = weight;
}

double Neuron::GetOutput() const {
	double sum = 0;

	for (size_t i = 0; i < inputsSize; i++)
		sum += inputs[i] * weights[i];

	return type == NeuronType::hidden ? ActivationFunction(sum) : sum;
}

double Neuron::GetDerivativeOutput() const {
	double sum = 0;

	for (size_t i = 0; i < inputsSize; i++)
		sum += inputs[i] * weights[i];

	return /*type == NeuronType::hidden ?*/ ActivationDerivativeFunction(sum)/* : 1*/;
}

void Neuron::Print() const {
	if (type == NeuronType::input) {
		cout << "InputNeuron: ";
	}
	else if (type == NeuronType::hidden) {
		cout << "HiddenNeuron: ";
	}
	else {
		cout << "OutputNeuron: ";
	}

	cout << "inputs: [ ";
	for (size_t i = 0; i < inputsSize; i++)
		cout << inputs[i] << " ";

	cout << "], weights: [ ";
	for (size_t i = 0; i < inputsSize; i++)
		cout << weights[i] << " ";

	double sum = 0;
	for (size_t i = 0; i < inputsSize; i++)
		sum += inputs[i] * weights[i];

	cout << "], out: " << GetOutput() << endl;
}