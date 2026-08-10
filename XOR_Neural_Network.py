import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s=sigmoid(x)
    return s*(1-s)

lr=0.01
epochs=100000
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

np.random.seed(42)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_neurons))

output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

for epoch in range(epochs):
    z1=np.dot(X,hidden_weights)+hidden_bias
    a1=sigmoid(z1)

    z2=np.dot(a1,output_weights)+output_bias
    a2=sigmoid(z2)

    output_error = -2 * (Y - a2) * sigmoid_derivative(z2)
    hidden_error = (output_error @ output_weights.T) * sigmoid_derivative(z1)

    d_w1 = X.T @ hidden_error
    d_b1 = np.sum(hidden_error, axis=0, keepdims=True)

    d_w2=a1.T@output_error
    d_b2=np.sum(output_error,axis=0,keepdims=True)

    hidden_weights=hidden_weights-lr*d_w1
    hidden_bias=hidden_bias-lr*d_b1

    output_weights=output_weights-lr*d_w2
    output_bias=output_bias-lr*d_b2

x1=int(input("Enter X1: "))
x2=int(input("Enter X2: "))
X=np.array([x1,x2])
z1=np.dot(X,hidden_weights)+hidden_bias
A=sigmoid(z1)
z2=np.dot(A,output_weights)+output_bias
result=sigmoid(z2)

print(result)