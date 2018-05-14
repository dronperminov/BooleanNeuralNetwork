compiler=g++
flags=-Wall
files=include/Neuron.cpp include/OneLayerNeuralNetwork.cpp

neurologic:
	$(compiler) $(flags) $(files) BooleanNeuralNetwork.cpp -o BooleanNeuralNetwork

neurointerpolation:
	$(compiler) $(flags) $(files) InterpolationNeuralNetwork.cpp -o InterpolationNeuralNetwork

clean:
	rm -rf *.exe