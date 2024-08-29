<p align="center">
  <img alt="micrograd_pp" src="https://raw.githubusercontent.com/parsiad/micrograd-pp/main/logo.png">
</p>

![](https://github.com/parsiad/micrograd-pp/actions/workflows/tox.yml/badge.svg)
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

## Examples

* [Train a simple feedforward neural network on MNIST to classify handwritten digits](https://nbviewer.org/github/parsiad/micrograd-pp/blob/main/examples/mnist.ipynb)
* [Learn an n-gram model to generate text](https://nbviewer.org/github/parsiad/micrograd-pp/blob/main/examples/n-gram.ipynb)
* [Train a decoder-only transformer to generate text](https://nbviewer.org/github/parsiad/micrograd-pp/blob/main/examples/transformer.ipynb)

## Features

* **Core**
  * ☒ Reverse-mode automatic differentiation (`.backward`)
  * ☒ GPU support
* **Layers**
  * ☒ BatchNorm1d
  * ☒ Dropout
  * ☒ Embedding
  * ☒ LayerNorm
  * ☒ Linear
  * ☒ MultiheadAttention
  * ☒ ReLU
  * ☒ Sequential
* **Optimizers**
  * ☐ Adam
  * ☒ Stochastic Gradient Descent (SGD)
