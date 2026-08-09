import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Weight and Bias Gradient
def bias_derivative(y_pred,y_true):
    n=len(y_true)
    gradient=0

    for i in range(n):
        gradient+=1/n*(y_pred[i]-y_true[i])

    return gradient

def weight_derivative(x,y_pred,y_true):
    n=len(y_true)
    gradient=np.zeros(x.shape[1])

    for i in range(n):
        gradient+=1/n*(y_pred[i]-y_true[i])*x[i]

    return gradient

#Setting up the dataframe
df=pd.read_csv("prices.csv")
x=np.array(df[["Distance","Time"]])
y=np.array(df["Price"])

#Initializing the weights, bias, lr and epochs
w=np.array([0.3,0.2])
b=0.4
lr=0.01
epochs=10000
m=len(y)

#Training the model
for epoch in range(epochs):
    z=x@w+b
    dw=weight_derivative(x,z,y)
    db=bias_derivative(z,y)
    temp_w=w-lr*dw
    temp_b=b-lr*db
    w=temp_w
    b=temp_b

#Testing the model
x1=float(input("Enter the distance: "))
x2=float(input("Enter the time: "))
X=np.array([x1,x2])
prediction=w@X+b
print(f"The price of the trip will be {prediction}")