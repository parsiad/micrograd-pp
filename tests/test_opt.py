import numpy as np
import pytest

import micrograd_pp as mpp


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    np.random.seed(0)
    yield


def test_mse():
    n = 10
    coef = np.random.randn(3, 1)
    coef_hat = np.random.randn(3, 1)
    x = np.random.randn(n, 3)
    ε = 0.0 * np.random.randn(n, 1)
    y = x @ coef + ε

    coef_hat_ = mpp.Parameter(coef_hat)
    x_ = mpp.Constant(x)
    y_ = mpp.Constant(y)

    opt = mpp.SGD(lr=0.1)
    for _ in range(150):
        y_pred_ = x_ @ coef_hat_
        mse = ((y_pred_ - y_) ** 2).sum() / n
        mse.backward(opt=opt)
        opt.step()

    np.testing.assert_allclose(coef, coef_hat)
