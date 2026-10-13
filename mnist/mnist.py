import numpy as np
import ssl
import time
from sklearn.datasets import fetch_openml
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
nn = neural_network([784, 63, 10])
initial_loss = nn.loss(nn.feed_forward(X_train), Y_train)
print(f"Initial Loss: {initial_loss:.4f}")

nn.gd(eta=1.0, epochs=500, x=X_train, y=Y_train, loss_out=50)

# evaluation
train_pred = np.argmax(nn.feed_forward(X_train)[-1], axis=1)
test_pred = np.argmax(nn.feed_forward(X_test)[-1], axis=1)

print(f"Train accuracy: {np.mean(train_pred == y[:60000]):.4f}")
print(f"Test acuracy: {np.mean(test_pred == y_test):.4f}")

assert nn.losses[-1] < initial_loss, "Loss did not decrease"
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Runtime: {execution_time:.6f} seconds")