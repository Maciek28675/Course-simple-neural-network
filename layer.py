import numpy as np
from neuron import Neuron

"""
    This class is used to represent a neural network layer
    which consits of multiple neurons.

"""

class Layer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        # Create a list of neurons and
        # specify how many inputs each neuron has
        self.neurons = [Neuron(number_of_inputs_per_neuron) for _ in range(number_of_neurons)]

    # This method is used to pass data through the network to produce an output
    def forward(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])
    
    # This method is used for the learning process.
    # It implements gradient descent algorithm.
    def backward(self, errors, learning_rate):
        deltas = []

        for i, neuron in enumerate(self.neurons):
            delta = errors[i] * neuron.sigmoid_derivative()
            neuron.update_weights(delta, learning_rate)
            deltas.append(delta)
        
        return np.dot(np.array([neuron.weights for neuron in self.neurons]).T, deltas)

