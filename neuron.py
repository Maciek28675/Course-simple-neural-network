import numpy as np

"""
    This class is used to represent a single neuron.
    - Each neuron has weights, bias, inputs and output
    - We initialize weights and bias with uniform random distribution.
    - Output is calculated by mutliplying vector of inputs with
      vector of weights, passing it through activation function and
      adding the bias
    - Delta: error
"""

class Neuron():
    def __init__(self, number_of_inputs):
        self.weights = np.random.uniform(size=number_of_inputs)
        self.bias = -1
    
    def activate(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(np.dot(inputs, self.weights) + self.bias)

        return self.output

    # ==== Activation functions ====

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        pass
    
    def sigmoid_derivative(self):
        return self.output * (1 - self.output)
    
    def relu_derivative(self):
        pass
    
    # ==============================

    def update_weights(self, delta, learning_rate):
        self.weights += learning_rate * delta * self.inputs
        self.bias += learning_rate * delta