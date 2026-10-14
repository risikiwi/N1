import ssl
import time
import json
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from processor import neural_network

start_time = time.perf_counter()
np.random.seed(0)
ssl._create_default_https_context = ssl._create_unverified_context
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y = mnist["data"], mnist["target"]
print(y.shape)

# preprocess
X = X/255.0
y = y.astype(int)
Y = np.eye(10)[y] # one hot encode

# split data
X_train, X_test = X[:60000], X[60000:]
Y_train, Y_test = Y[:60000], Y[60000:]
y_test = y[60000:]

# build and train
nn = neural_network([784, 64, 10])
initial_loss = nn.loss(nn.feed_forward(X_train), Y_train)
print(f"Initial Loss: {initial_loss:.4f}")

nn.gd(eta=1.0, epochs=50, x=X_train, y=Y_train, batch_size = 128, loss_out=5)
print(nn.losses[45], nn.losses[-1])
print(f"Relative change over last 5 epochs: {(nn.losses[45] - nn.losses[-1]) / nn.losses[45]:.2%}")
# plt.plot(nn.losses, label='Training Loss')
# plt.yscale("log") # to show slow decline better, not very visible
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.show()

# evaluation
train_pred = np.argmax(nn.feed_forward(X_train)[-1], axis=1)
test_pred = np.argmax(nn.feed_forward(X_test)[-1], axis=1) # a.shape = (# of layer, # of input per layer, # of neurons)

print(f"Train accuracy: {np.mean(train_pred == y[:60000]):.4f}")
print(f"Test acuracy: {np.mean(test_pred == y_test):.4f}")

assert nn.losses[-1] < initial_loss, "Loss did not decrease"
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Runtime: {execution_time:.6f} seconds")

def export_tests(inputs, predictions, path="tests.json"):
    with open(path, "w") as f:
        json.dump({
            "inputs" : [list(map(float, x)) for x in inputs],
            "predictions" : [int(p) for p in predictions],
        }, f)

export_tests(X_test[:5], test_pred[:5])
nn.export_network(["sigmoid", "sigmoid"])