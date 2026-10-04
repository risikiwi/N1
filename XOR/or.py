import numpy as np
# recreating the and gate before moving up to xor, with 1 neuron

x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y_and = np.array([[0], [1], [1], [1]], dtype=float)

print(x.shape)
print(y_and.shape)

np.random.seed(0)
# weights start randomized to make different neurons learn different features
# techinically, with 1 neuron there's no symmetry to break, so any arbitrary value (reasonably small) will work
w = np.random.randn(2, 1) * 0.1
# biases start @ zero to prevent accidental directional skew before training
b = np.zeros((1, 1))

print(w)
print(b)

losses = np.array([])

def pre_activation():
    return x@w + b

def sigmoid(z):
    return 1.0/ (1+np.exp(-z))

def loss(a):
    return -np.mean(y_and*np.log(a) + (1-y_and)*np.log(1-a))

def backprop(a):
    dL_dz = a-y_and
    dL_db = np.mean(dL_dz, axis=0, keepdims=True)
    dL_dw = x.T@dL_dz / x.shape[0]
    return dL_dw, dL_db

def gd(eta, epochs):
    global w, b
    for epoch in range(epochs):
        z = pre_activation()
        a = sigmoid(z)
        l = loss(a)
        np.append(losses, l)
        dL_dw, dL_db = backprop(a)
        w -= eta*dL_dw
        b -= eta*dL_db

        if(epoch % 200 == 0):
            print(l)

    return a
    
a = gd(0.05, 1000)
print(np.column_stack((y_and, a.round())))





