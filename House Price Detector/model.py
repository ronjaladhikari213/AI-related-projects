# Importing the Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gradient Descent and Its Derivative


def bias_derivative(y_true, y_pred):
    n = len(y_true)
    gradient = 0

    for i in range(n):
        gradient += -(2 / n) * (y_true[i] - y_pred[i])

    return gradient


def weight_derivative(x, y_true, y_pred):
    n = len(y_true)
    gradient = 0

    for i in range(n):
        gradient += -(2 / n) * x[i] * (y_true[i] - y_pred[i])

    return gradient


# Organizing the Dataset
df = pd.read_csv("Housing.csv")
X = df["area"]
Y = df["price"]
x_mean = X.mean()
x_std = X.std()
X = (X-x_mean)/x_std

# Initialzing Weight, Bias, Learning Rate and No. of Iterations
w = 0
b = 0
lr = 0.01
epochs = 10000

# Trainign the Model
for epoch in range(epochs):
    prediction = w*X+b
    d_costw = weight_derivative(X, Y, prediction)
    d_costb = bias_derivative(Y, prediction)
    temp_w = w-lr*d_costw
    temp_b = b-lr*d_costb
    w = temp_w
    b = temp_b
prediction=w*X+b

# Plotting the Graph
plt.scatter(X, Y, label="Data points")
plt.plot(X, prediction, color="black", label="Straight line")
plt.xlabel("Area of the House ->")
plt.ylabel("Price of the House->")
plt.legend()
plt.show()

#Testing the Data
x=float(input("Enter the area of the house: "))
x=(x-x_mean)/x_std
pred=w*x+b
print(f"The price of the house is: {pred}")
