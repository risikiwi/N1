import numpy as np

# 1 neuron cannot make an XOR as that would lead to
# w2(1) > b, w1(1) > b, w1(1) + w2(1) <= b, contradiction
sizes = [2, 2, 1]

x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)

np.random.seed(0)

w = [np.random.randn(i, n) * 0.1 for i, n in zip(sizes[:-1], sizes[1:])]
b = [np.zeros((1, n)) for n in sizes[1:]]

losses = []


# spits out vector of pre-activation of each neuron of one layer
def pre_activation(x, w, b):
    return x@w + b

# spits out vector of sigmoid of each neuron of one layer
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# average loss
def loss(a):
    return -np.mean((y * np.log(a[-1]) + (1.0 - y) * np.log(1.0 - a[-1])))

def backprop(a, w):
    n = x.shape[0]
    dz = a[-1] - y
    dw = [None] * len(w)
    db = [None] * len(b)
    for layer in reversed(range(len(w))):
        dw[layer] = a[layer].T @ dz/n
        db[layer] = np.mean(dz, axis=0, keepdims=True)
        if layer > 0: # back to front layer, calculate dw, db based on output
            da = dz @ w[layer].T #
            dz = da * a[layer] * (1-a[layer]) # dL/dz2 * dz2/da1 * da1/dz1
    return dw, db

def gd(eta, epochs):
    global w, b
    for epoch in range(epochs):
        a = [x]
        # each a dependent on the outputs from prev layer
        for layer in range(len(sizes)-1):
            z = pre_activation(a[layer], w[layer], b[layer]) # a[-1]
            a.append(sigmoid(z))
        l = loss(a)
        losses.append(l)
        dw, db = backprop(a, w)
        for layer in range(len(sizes)-1):
            w[layer] -= eta * dw[layer]
            b[layer] -= eta * db[layer]
        if(epoch % 200 == 0):
            print(l)

    return a[-1]

a = gd(5, 2000)
print(np.column_stack((y, a)))

# values for eta, epochs:
# 0.5, 2000 - eta is too small, loss reduces rlly slowly
# values for eta, epochs:
# 1, 2000 - eta is good size, loss reduced significantly in last 600 epochs
# 2, 2000 - eta is big, loss reduced significantly after 800 epochs
# 10, 2000 - eta is too big, kept overshooting and the loss barely reduces