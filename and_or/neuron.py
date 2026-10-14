# w = [0.5, -1, 2] x = [1, 2, 3] b = 0.5 
# compute z, ReLU vs Sigmoid by hand & compare with program
import numpy as np
# w = (D, N)
# x = (D, I)

w = np.array([0.5, -1, 2])
x = np.array([1, 2, 3])
b = -5.5


def pre_activation(w, x, b):
    return w@x+b # dot product w/ 1d arrays

def sigmoid(pa):
    return np.power(1+np.exp(-pa),-1)

def ReLU(pa):
    return max(0, pa)


p_a = pre_activation(w, x, b)
print(sigmoid(p_a))
print(ReLU(p_a))

