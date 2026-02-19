import pytest

import micrograd_pp as mpp

np = mpp.numpy


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    np.random.seed(0)
    yield


@pytest.mark.parametrize(
    ("opt_factory", "num_steps", "atol"),
    [
        (lambda: mpp.SGD(lr=0.1), 150, 1e-8),
        (lambda: mpp.AdamW(lr=0.2, weight_decay=0.0), 600, 1e-8),
    ],
    ids=("sgd", "adamw"),
)
def test_mse(opt_factory, num_steps: int, atol: float):
    n = 10
    coef = np.random.randn(3, 1)
    coef_hat = np.random.randn(3, 1)
    x = np.random.randn(n, 3)
    ε = 0.0 * np.random.randn(n, 1)
    y = x @ coef + ε

    coef_hat_ = mpp.Parameter(coef_hat)
    x_ = mpp.Constant(x)
    y_ = mpp.Constant(y)

    opt = opt_factory()
    for _ in range(num_steps):
        y_pred_ = x_ @ coef_hat_
        mse = ((y_pred_ - y_) ** 2).sum() / n
        mse.backward(opt=opt)

    np.testing.assert_allclose(coef, coef_hat, rtol=0.0, atol=atol)
