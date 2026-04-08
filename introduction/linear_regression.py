import numpy as np

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

print(f"\nw = {w}, b = {b}")
print("Prediction: x = 20 -> " + str(predict(20,w,b)))