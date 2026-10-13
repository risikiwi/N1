# MNISt from scratch

Brief Description: a NumPy neural network (no ML frameworks) trained over MNIST.

## How to run
pip install numpy scikit-learn
python mnist.py

## Architecture
784 - 64 - 10, sigmoid activations, cross-entropy loss, full-batch gradient descent

## Results
| Hidden size | Learning rate | Epochs | Time | Train acc | Test acc |
|-------------|---------------|--------|------|-----------|----------|
| 64          | 1.0           | 500    | 40s  | 0.930     | 0.932    |
| 64          | 1.5           | 500    | 37s  | 0.938     | 0.937    |
| 64          | 1.0           | 500    | 40s  | 0.930     | 0.932    |
| 64          | 1.0           | 500    | 40s  | 0.930     | 0.932    |
| 64          | 1.0           | 500    | 40s  | 0.930     | 0.932    |

## How I chose the learning rate
Learning rate = 1.0
Appearnace - Initial reults showed noise in noise in test accuracy. Typically a network like this can reach roughly 97-98%.
Analyze - Gap b/w expected & output (97-93) indicates the model might requires more training
Conclusion - Increase learning rate by 0.5
