compiler=g++
flags=-Wall
files=include/Neuron.cpp include/OneLayerNeuralNetwork.cpp

neurologic:
	$(compiler) $(flags) $(files) BooleanNeuralNetwork.cpp -o BooleanNeuralNetwork

clean:
	rm -rf *.exe