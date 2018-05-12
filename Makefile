compiler=g++
flags=-Wall
files=include/Neuron.cpp
name=BooleanNeuralNetwork

neurologic:
	$(compiler) $(flags) $(files) BooleanNeuralNetwork.cpp -o $(name)

clean:
	rm -rf $(name)