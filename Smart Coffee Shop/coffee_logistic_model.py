# For Logistic Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid Function


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Setting up the dataset
df = pd.read_csv("smart_coffee_shop_pricing_dataset_realistic.csv")
df["No. of Customers"] = (df["No. of Customers"] > 100).astype(int)
X = df[["Coffee Price", "Rain", "Weekend"]].values

# Target
Y = np.array(df["No. of Customers"])

# Initialzing weights, learning rate and bias
w = np.array([1, 1.5, -2])
b = 2.5
lr = 0.01
epochs = 10000
m = len(Y)
lambda_=10
for epoch in range(epochs):
    z = X@w+b
    y_pred=sigmoid(z)
    dw = (1 / m) * (X.T @ (y_pred - Y)) + (lambda_ / m) * w
    db = (1 / m) * np.sum(y_pred - Y)
    w = w-lr*dw
    b = b-lr*db

# Testing the Model
l = []
inp1 = float(input(("What is the price: ")))
l.append(inp1)
inp2 = int(input("Hit 1 for rain, and 0 for no rain: "))
l.append(inp2)
inp3 = int(input("Hit 1 for weekend and 0 for no weekend: "))
l.append(inp3)

l=np.array(l)

z = l@w+b
y_pred=sigmoid(z)
print("Probability:", y_pred)
print("More than 100 customers:", y_pred >= 0.5)
