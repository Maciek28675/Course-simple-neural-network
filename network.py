import numpy as np
from layer import Layer

"""
    This class is used to initialize a neural network
    with given number of layers and train it
"""

class Network():
    def __init__(self, layers, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []
        self.mse_per_layer = [[] for _ in range(len(layers) - 1)]

        # Initialize layers
        for i in range (len(layers) - 1):
            self.layers.append(Layer(layers[i+1], layers[i]))
    
    # TODO: zwrocic tablice mse zeby moc zrobic wykresy
    def train(self, inputs, outputs):
        for epoch in range(self.epochs):
            total_error = 0
            temp_mse = [0] * len(self.layers)
            
            for x, y in zip(inputs, outputs):
                # Forward pass
                activations = [x]

                for layer in self.layers:
                    activations.append(layer.forward(activations[-1]))
                
                # Calculate error
                output_errors = y - activations[-1]
                total_error += np.sum(output_errors ** 2)

                for i, activation in enumerate(activations[1:], 1):
                    temp_mse[i - 1] += np.mean((activation - y) ** 2)

                # Backward pass
                errors = output_errors
                
                for i in reversed(range(len(self.layers))):
                    errors = self.layers[i].backward(errors, self.learning_rate)

            # Save calculated mse in history so that you can plot it later
            for i in range(len(self.layers)):
                self.mse_per_layer[i].append(temp_mse[i] / len(inputs))

            mse = total_error / len(inputs)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, MSE: {mse}')

            if(mse < 0.005):
                print(f"Goal mse achieved. Terminating training at epoch {epoch}")
                break

    # The purpose of this method is to use a trained model on new data
    def predict(self, inputs):
        activations = inputs

        for layer in self.layers:
            activations = layer.forward(activations)
        
        for i in range(len(activations)):
            if activations[i] > 0.5:
                activations[i] = 1
            else:
                activations[i] = 0

        return activations