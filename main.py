import numpy as np
from network import Network

"""
    XOR:

    a | b | Output
   ---------------
    0 | 0 |   0  
    0 | 1 |   1
    1 | 0 |   1
    1 | 1 |   0

"""

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# 2 input neurons, 2 in hidden layer and 1 output
layers = [2, 2, 1]
lr = 0.4
epochs = 10000

# TODO: Przerobić tak, zeby dalo sie tworzyc siec modulowo
#       tj. Warstwa(8), f. aktywacji, Wartswa(8), itd...
# TODO: Zaimplementowac momentum, adaptive lr i mini batch

neural_network = Network(layers, lr, epochs)
neural_network.train(inputs, outputs)

predicted_output = np.array([neural_network.predict(x) for x in inputs])
print("Predicted Output:\n", predicted_output)