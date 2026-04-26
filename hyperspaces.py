#%% 

import numpy as np 
#from matplotlib import pyplot as plt

X1, X2, X3, y = np.loadtxt("pizza_2.txt", skiprows=1, unpack=True)

X = np.column_stack((X1, X2, X3))
Y = y.reshape(-1, 1)


def predict(X1,X2,X3,w1,w2,w3,b):
    return X1*w1+X2*w2+X3*w3+b

def loss(X1,X2,X3,w1,w2,w3,b,Y):
    return np.average((predict(X1,X2,X3,w1,w2,w3,b)-Y)**2)



#%%