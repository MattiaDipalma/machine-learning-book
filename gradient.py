# %%

import numpy as np
from matplotlib import pyplot as plt
#adds a visualisation of the code

def predict(X,w,b):
    return X*w+b

def loss(X,Y,w,b):
    return np.average((predict(X,w,b)-Y)**2)

def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)

def train(X, Y, iterations, lr):
    w = b =  0
    for i in range(iterations):
        print(f"Iteration {i} => Loss: {loss(X,Y,w,0)}")
        print(f"Iteration {i} => Loss: {loss(X,Y,w,b)}")
        w_gradient, b_gradient = gradient(X, Y, w, b)

        w -= w_gradient * lr
        b -= b_gradient * lr 

        if abs(w_gradient) < 1e-8 and abs(b_gradient) < 1e-8:
            break
    return w, b

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=10000, lr=0.001) 
p = 20
print(f"\nw={w}, b={b}")
print(f"Prediction: x={p} => y={predict(p, w, b)}")

def linear_regression(x):
    return w*x+b

x = np.linspace(0,max(X),500)

plt.plot(x,linear_regression(x),color = 'g',label="trend")
plt.plot(X,Y,'o',color='r',label="real data")
plt.plot(p,predict(p,w,b),'o',color='b',label="predicted data")

plt.title("Lineare Regression mit Gradient Verfahren")
plt.xlabel("reservations")
plt.ylabel("pizzas")
plt.legend()
plt.grid(True)
plt.show()
# %%
