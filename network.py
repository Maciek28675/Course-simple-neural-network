from layer import Layer

"""
    This class is used to initialize a neural network
    with given number of layers
"""

class Network():
    def __init__(self, layers, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []

        # Initialize layers
        for i in range (len(layers) - 1):
            self.layers.append(Layer(layers[i+1], layers[i]))