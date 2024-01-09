import numpy as np
import pytest

import micrograd_pp as mpp


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    np.random.seed(0)
    yield


def cross_entropy_loss(input_: mpp.Expr, target: mpp.Expr) -> mpp.Expr:
    n, _ = input_.shape
    input_max = input_.max(dim=1)
    delta = input_ - input_max.expand(input_.shape)
    log_sum_exp = delta.exp().sum(dim=1).log().squeeze()
    return (log_sum_exp - delta[np.arange(n), target]).sum() / n


def test_mnist(batch_sz: int = 64, n_epochs: int = 3):
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

    assert accuracy >= 0.9
