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
| 64          | 1.5           | 2000   | 258s | 0.968     | 0.961    |

| Hidden size | Learning rate | Epochs | Batch Size | Time | Train acc | Test acc |
|-------------|---------------|--------|------------|------|-----------|----------|
| 64          | 1.0           | 100    | 100        | 54s  | 0.999     | 0.973    |
| 64          | 1.0           | 50     | 128        | 31s  | 0.998     | 0.974    |

## How I chose the hyperparameters
Learning rate = 1.0
Appearnace - Initial reults showed noise in noise in test accuracy. Typically a network like this can reach roughly 97-98%.
Analyze - Gap b/w expected & output (97-93) indicates the model might requires more training
Conclusion - Increase learning rate by 0.5

Learning rate = 1.5
Appearnace - Initial reults showed noise in noise in test accuracy. Typically a network like this can reach roughly 97-98%.
Additional - plots nn.losses 
<img src="https://i.postimg.cc/C5pvptsQ/Screenshot-2026-10-13-at-13-58-46.png" alt="Displays an image of training loss" width="400">
Analyze - Gap b/w expected & output (97-94) indicates the model requires more training. Training is slow, however.
Conclusion - Run 2000 epochs. Use log on yscale show slow decline better on the plot, not very visible currently.

Learning rate = 1.5, epoch = 2000
Appearnace - Network Reaches Expected Accuracy. Training is slow and took 4 minutes.
<img src="https://i.postimg.cc/Zng6hymK/Screenshot-2026-10-13-at-14-13-47.png" alt="Displays an image of training loss" width="400">
Analyze - Full-batch Gradient descent is inefficient. 1 weight update per epoch. 2000 epochs = 2000 updates.
Conclusion - Edit gd in processor.py to train with batches of different sizes to obtain many more updates per epoch

Learning rate = 1.0, epoch = 100, batch size = 100
Appearnace - Network Reaches Expected Accuracy on testing. Higher than expected on Training
<img src="https://i.postimg.cc/9QvRFjsP/Screenshot-2026-10-13-at-14-30-42.png" alt="Displays an image of training loss" width="400">
Analyze - Potential overfitting on testing data
Conclusion - Less epochs, larger batch_size to decrease # of updates.

Learning rate = 1.0, epoch = 50, batch size = 128
Appearnace - Network Reaches Expected Accuracy on testing. Higher than expected on Training, but improved slightly
Analyze - Potential overfitting on testing data
Conclusion - Less epochs, larger batch_size to decrease # of updates.

## What I learned
One hot encoding - turn numeric labels into categories for training
Underfit vs Overfit
- Underfitting when network doesn't do well on either training nor testing data. Occurs when the chosen algorithmn is too SIMPLE for the dataset
- Overffiting when network does well on training but not testing data. Occurs when the chosen algorithmn is too COMPLICATED for the dataset
- Bias - missing features, simple assumptions
- Variance - extreme sensitivity to overall irrelevant noise
Train in batch. Instead of running a ton of epochs and updating params every epoch---update params in batches per epoch. Speeding up the 
If batches are in the same order, network can pick up on the ordering and overfit.

## Next Steps
Mini-batches, generalize to other datasets
Add function to automatically attempt various epochs/ learning rate & optimize hyperparameters
Website interactive

