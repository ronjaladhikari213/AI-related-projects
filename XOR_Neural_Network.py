import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

np.random.seed(42)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

hidden_weigts = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_neurons))

output_weigts = np.random.uniform(size=(output_neurons, hidden_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))
