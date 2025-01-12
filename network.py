import numpy as np
import matplotlib.pyplot as plt
from layer import Layer

"""
    This class is used to initialize a neural network
    with given number of layers and train it
"""

class Network():
    def __init__(self, layers, learning_rate=0.1, epochs=1000):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []
        self.learning_rate_history = []

        # Initialize layers
        for i in range (len(layers) - 1):
            self.layers.append(Layer(layers[i+1], layers[i]))
    
    # Step Decay: Reduce learning rate by factor 'drop' every 'epochs_drop'
    def adjust_learning_rate(self, epoch):
        drop = 0.5
        epochs_drop = 10
        self.learning_rate = self.initial_learning_rate * np.power(drop, np.floor(epoch / epochs_drop))
        self.learning_rate_history.append(self.learning_rate)

    # TODO: zwrocic tablice mse zeby moc zrobic wykresy
    def train(self, inputs, outputs):
        for epoch in range(self.epochs):
            self.adjust_learning_rate(epoch)
            total_error = 0

            for x, y in zip(inputs, outputs):
                # Forward pass
                activations = [x]

                for layer in self.layers:
                    activations.append(layer.forward(activations[-1]))
                
                # Calculate error
                output_errors = y - activations[-1]
                total_error += np.sum(output_errors ** 2)

                # Backward pass
                errors = output_errors
                
                for i in reversed(range(len(self.layers))):
                    errors = self.layers[i].backward(errors, self.learning_rate)

            if epoch % 100 == 0:
                mse = total_error / len(inputs)
                print(f'Epoch {epoch}, MSE: {mse}, Learning Rate: {self.learning_rate}')
        
        # Plot learning rate history after training
        self.plot_learning_rate()

    def plot_learning_rate(self):
        plt.figure()
        plt.plot(self.learning_rate_history)
        plt.title("Learning Rate Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid()
        plt.show()  

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