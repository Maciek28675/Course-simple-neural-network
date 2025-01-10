import numpy as np
from network import Network

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# 2 input neurons, 2 in hidden layer and 1 output
layers = [2, 2, 1]
lr = 0.1
epochs = 10000

neural_network = Network(layers, lr, epochs)
neural_network.train(inputs, outputs)

predicted_output = np.array([neural_network.predict(x) for x in inputs])
print("Predicted Output:\n", predicted_output)