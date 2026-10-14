import numpy as np


class neural_network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.layers = len(sizes)
        self.biases = [np.zeros((1, n)) for n in sizes[1:]]
        self.weights = [np.random.randn(i, n)*0.1 for i, n in zip(sizes[:-1], sizes[1:])]
        self.losses = []


    @staticmethod
    def loss(a, y):
        return -np.mean((y * np.log(a[-1]) + (1.0 - y) * np.log(1.0 - a[-1])))


    @staticmethod
    def pre_activation(x, w, b):
        z = x@w+b
        return z


    @staticmethod
    def sigmoid(z):
        return 1.0/ (1+np.exp(-z))


    def feed_forward(self, x):
        a = [x]
        for layer in range(self.layers - 1):
            z = self.pre_activation(a[layer], self.weights[layer], self.biases[layer])
            a.append(self.sigmoid(z))
        return a


    def backprop(self, a, x, y):
        n = x.shape[0]
        dw = [None] * len(self.weights)
        db = [None] * len(self.biases)
        dz = a[-1] - y
        for layer in reversed(range(self.layers - 1)):
            dw[layer] = a[layer].T @ dz / n
            db[layer] = np.mean(dz, axis=0, keepdims = True)
            if layer > 0:
                da = dz @ self.weights[layer].T
                dz = da * a[layer] * (1-a[layer]) # dL/dz2 * dz2/da1 * da1/dz1
        return dw, db

    
    def gd(self, eta, epochs, x, y, batch_size=None, loss_out=0):
        n = x.shape[0]
        if batch_size is None:
            batch_size = n
        w = self.weights
        b = self.biases
        for epoch in range(epochs):
            idx = np.random.permutation(n) 
            x_s, y_s = x[idx], y[idx] # shuffled data
            batch_losses = []
            # updates weights, biases by batch
            for start in range(0, n, batch_size):
                xb = x_s[start:start + batch_size]
                yb = y_s[start:start + batch_size]
                a = self.feed_forward(xb)
                dw, db = self.backprop(a, xb, yb)
                for layer in range(self.layers-1):
                    w[layer] -= eta * dw[layer]
                    b[layer] -= eta * db[layer]
                batch_losses.append(self.loss(a, yb))
            l = np.mean(batch_losses)
            self.losses.append(l)
            if loss_out and (epoch + 1) % loss_out == 0:
                print(f"Epoch {epoch+1}: loss = {l:.4f}")
        return a[-1]








