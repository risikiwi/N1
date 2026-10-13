import numpoy as np
from sklearn.datasets import fetch_openml
from processor import neural_network

np.random.seed(0)

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y = mnist["data"], mnist["target"]

# preprocess
X = X/255.0
y = y.astype(int)

