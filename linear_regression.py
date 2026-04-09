# %%

import numpy as np
from matplotlib import pyplot as plt
#adds a visualisation of the code

def predict(X,w,b):
    return X*w+b

def loss(X,Y,w,b):
    return np.average((predict(X,w,b)-Y)**2)

def train(X,Y,iterations,lr):
    w=b=0
    for i in range(iterations):
        current_loss = loss(X,Y,w,b)
        print(f"Iteration {i} -> Loss: {current_loss}")

        if loss(X,Y,w+lr,b)<current_loss:
            w+=lr
        elif loss(X,Y,w-lr,b)<current_loss:
            w-=lr
        elif loss(X,Y,w,b+lr)<current_loss:
            b+=lr
        elif loss(X,Y,w,b-lr)<current_loss:
            b-=lr
        else:
            return w,b
        
    raise Exception(f"Couldn't converge within {iterations} iterations")

#train(10,10,10,10)

X,Y = np.loadtxt("pizza.txt",skiprows=1, unpack=True)
w,b = train(X,Y, iterations =10000, lr=0.01)
p=20

print(f"\nw = {w}, b = {b}")
print(f"Prediction: x = {p} -> " + str(predict(p,w,b)))

def linear_regression(x):
    return w*x+b

x = np.linspace(0,max(X),500)

plt.plot(x,linear_regression(x),color = 'g',label="trend")
plt.plot(X,Y,'o',color='r',label="real data")
plt.plot(p,predict(p,w,b),'o',color='b',label="predicted data")

plt.title("Lineare Regression")
plt.xlabel("reservations")
plt.ylabel("pizzas")
plt.legend()
plt.grid(True)
plt.show()
# %%
