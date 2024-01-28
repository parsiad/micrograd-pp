<p align="center">
  <img alt="micrograd_pp" src="https://raw.githubusercontent.com/parsiad/micrograd-pp/main/logo.png">
</p>

<a href="https://github.com/parsiad/micrograd-pp"><img alt="GitHub" src="https://img.shields.io/badge/github-%23121011.svg?logo=github"></a>

Micrograd++ is a minimalistic wrapper around NumPy which adds support for automatic differentiation.
Designed as a learning tool, Micrograd++ provides an accessible entry point for those interested in understanding automatic differentiation and backpropagation or seeking a clean, educational resource.

Micrograd++ draws inspiration from Andrej Karpathy's awesome [micrograd](https://github.com/karpathy/micrograd) library, prioritizing simplicity and readability over speed.
Unlike micrograd, which tackles scalar inputs, Micrograd++ supports tensor inputs (specifically, NumPy arrays).
This makes it possible to train larger networks.

## Usage

Micrograd++ is not yet pip-able.
Therefore, you will have to clone the Micrograd++ repository to your home directory and include it in any script or notebook you want to use it in by first executing the snippet below:

```python
import sys
sys.path.insert(0, os.path.expanduser("~/micrograd-pp/python"))
```

## Example: MNIST

![](https://upload.wikimedia.org/wikipedia/commons/f/f7/MnistExamplesModified.png)

[MNIST](https://en.wikipedia.org/wiki/MNIST_database) is a dataset of handwritten digits (0-9) commonly used for training and testing image processing systems.
It consists of 28x28 pixel grayscale images, with a total of 60,000 training samples and 10,000 test samples.
It's widely used in machine learning for digit recognition tasks.

Below is an example of using Micrograd++ to train a simple [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) on MNIST.

```python
import micrograd_pp as mpp
import numpy as np


batch_sz = 64
n_epochs = 3

mnist = mpp.datasets.load_mnist(normalize=True)
train_images, train_labels, test_images, test_labels = mnist

# Flatten images
train_images = train_images.reshape(-1, 28 * 28)
test_images = test_images.reshape(-1, 28 * 28)

# Drop extra training examples
trim = train_images.shape[0] % batch_sz
train_images = train_images[: train_images.shape[0] - trim]

# Shuffle
indices = np.random.permutation(train_images.shape[0])
train_images = train_images[indices]
train_labels = train_labels[indices]

# Make batches
n_batches = train_images.shape[0] // batch_sz
train_images = np.split(train_images, n_batches)
train_labels = np.split(train_labels, n_batches)

# Optimizer
opt = mpp.SGD(lr=0.01)

# Feedforward neural network
model = mpp.Sequential(
    mpp.Linear(28 * 28, 128),
    mpp.ReLU(),
    mpp.Linear(128, 10),
)


def cross_entropy_loss(input_: mpp.Expr, target: mpp.Expr) -> mpp.Expr:
    n, _ = input_.shape
    input_max = input_.max(dim=1)
    delta = input_ - input_max
    log_sum_exp = delta.exp().sum(dim=1).log().squeeze()
    return (log_sum_exp - delta[np.arange(n), target]).sum() / n


# Train
accuracy = float("nan")
for epoch in range(n_epochs):
    for batch_index in np.random.permutation(np.arange(n_batches)):
        x = mpp.Constant(train_images[batch_index])
        y = train_labels[batch_index]
        loss = cross_entropy_loss(model(x), y)
        loss.backward(opt=opt)
        opt.step()
    test_x = mpp.Constant(test_images)
    test_fx = model(test_x)
    pred_labels = np.argmax(test_fx.value, axis=1)
    accuracy = (pred_labels == test_labels).mean().item()
    print(f"Test accuracy at epoch {epoch}: {accuracy * 100:.2f}%")
```
