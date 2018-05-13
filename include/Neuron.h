#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

// тип нейрона
enum class NeuronType {
	input,
	hidden,
	output
};

typedef double (*ActivationPointer)(double); // тип указателя на вещественную функцию вещественного аргумента

double Sigmoid(double x); // сигмоидальная функция активации
double SigmoidDerivative(double x); // производная сигмоидальной функции активации

double HiperbolicTangent(double x); // гиперболический тангенс
double HiperbolicTangentDerivative(double x); // производная гиперболического тангенса

double ReLU(double x); // выпрямитель
double ReLUDerivative(double x); // производная выпрямителя

class Neuron {
	size_t inputsSize; // число входов нейрона
	NeuronType type; // тип нейрона
	std::vector<double> inputs; // входные сигналы
	std::vector<double> weights; // весы на связи

	ActivationPointer ActivationFunction; // указатель на функцию активации
	ActivationPointer ActivationDerivativeFunction; // указатель на производную функции активации

public:
	Neuron(NeuronType type, size_t inputsSize = 1); // по-умолчанию у нейрона один вход
	Neuron(NeuronType type, size_t inputsSize, ActivationPointer ActivationFunction, ActivationPointer ActivationDerivativeFunction);

	std::vector<double> GetInputs() const;
	void SetInputs(const std::vector<double>& inputs);

	std::vector<double> GetWeights() const;
	void SetWeights(const std::vector<double>& weights);

	double GetInput(size_t index) const;
	void SetInput(size_t index, double input);

	double GetWeight(size_t index) const;
	void SetWeight(size_t index, double weight);

	double GetOutput() const;
	double GetDerivativeOutput() const;
	void Print() const;
};