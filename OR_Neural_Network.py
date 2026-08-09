# Building Neural Network to represent OR gate
import numpy as np

# Training Dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0],
              [1],
              [1],
              [1]])

# Setting up The random value of Weights and Bias
np.random.seed(42)
weights = np.random.randn(2, 1)
bias = np.random.randn(1)

# Sigmoid and Derivative of sigmoid activation function for value to be [0,1]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    # It is designed from the derivated form, the mathematical derivative form looks a little different
    return x*(1-x)


# iterations
lr = 0.1  # Keeping this low will help the model learn more steadier way but the values of it may vary
epochs = 100000

# Training the model
for epoch in range(epochs):
    z = np.dot(X, weights)+bias
    output = sigmoid(z)

    error = Y-output

    delta = error*sigmoid_derivative(output)
    weights += lr*np.dot(X.T, delta)
    bias += lr*np.sum(delta)

# Testing the Model
print("Prediction:")
pred = np.dot(X, weights)+bias
result = sigmoid(pred)
print(result)
