#For Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Setting up the dataset
df=pd.read_csv("smart_coffee_shop_pricing_dataset_realistic.csv")
X = df[["Coffee Price", "Rain", "Weekend"]].values

#Target
Y=np.array(df["No. of Customers"])

#Initialzing weights, learning rate and bias
w=np.array([1,1.5,-2])
b=2.5
lr=0.01
epochs=10000
lambda_=10
m=len(Y)

for epoch in range(epochs):
    y_pred=X@w+b
    dw=(X.T @ (y_pred - Y)) / m + (lambda_ / m) * w
    db= np.sum(y_pred - Y) / m
    w=w-lr*dw
    b=b-lr*db

#Testing the Model
l=[]
inp1=float(input(("What is the price: ")))
l.append(inp1)
inp2=int(input("Hit 1 for rain, and 0 for no rain: "))
l.append(inp2)
inp3=int(input("Hit 1 for weekend and 0 for no weekend: "))
l.append(inp3)
l=np.array(l)

y_pred=l@w+b
print("The No. of Customers is: ",y_pred)