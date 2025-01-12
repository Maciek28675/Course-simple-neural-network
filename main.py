import numpy as np
import matplotlib.pyplot as plt

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

print(np.random.randn(2) * np.sqrt(1 / 2))
# 2 input neurons, 2 in hidden layer and 1 output
layers = [2, 2, 1]
lr = 0.1
epochs = 15000

# TODO: Przerobić tak, zeby dalo sie tworzyc siec modulowo
#       tj. Warstwa(8), f. aktywacji, Wartswa(8), itd...
# TODO: Zaimplementowac momentum, adaptive lr i mini batch

net = Network(layers, lr, epochs)
net.train(inputs, outputs)

predicted_output = np.array([net.predict(x) for x in inputs]) 
print("Predicted Output:\n", predicted_output)


# Plot mse for each layer
for i, mse_layer in enumerate(net.mse_per_layer):
   epochs_list = range(1, len(mse_layer)+1)
   plt.plot(epochs_list, mse_layer, label=f'Warstwa {i + 1}')

plt.title("MSE w poszczególnych warstwach")
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.show()