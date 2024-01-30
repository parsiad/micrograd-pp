import itertools
from typing import Generator

import numpy as np
import pytest

import micrograd_pp as mpp

DIMS = [0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2), None]


@pytest.fixture(autouse=True)
def run_before_and_after_tests() -> Generator[None, None, None]:
    np.random.seed(0)
    yield


def test_add() -> None:
    a = np.random.randn(4, 1, 2)
    b = np.random.randn(3, 2)
    a_ = mpp.Parameter(a)
    b_ = mpp.Parameter(b)
    c_ = a_ + b_
    c_.backward()
    np.testing.assert_equal(a_.grad, np.full_like(a, 3.0))
    np.testing.assert_equal(b_.grad, np.full_like(b, 4.0))


def test_add_scalar() -> None:
    a = np.random.randn(3, 2)
    a_ = mpp.Parameter(a)
    c = 2.0
    b_ = c + a_
    b_.backward()
    grad = np.ones_like(a)
    np.testing.assert_equal(a_.grad, grad)


def test_exp() -> None:
    a = np.random.randn(3, 2)
    a_ = mpp.Parameter(a)
    b_ = a_.exp()
    b_.backward()
    np.testing.assert_equal(a_.grad, b_.value)


@pytest.mark.parametrize("dim", DIMS)
def test_expand(dim: int | tuple[int, ...] | None) -> None:
    expand_shape = [4, 3, 2]
    shape = expand_shape.copy()
    if dim is None:
        dim = (0, 2)
    if isinstance(dim, int):
        dim = (dim,)
    n_copies = 1
    for d in dim:
        n_copies *= shape[d]
        shape[d] = 1
    a = np.random.randn(shape[0], shape[1], shape[2])
    a_ = mpp.Parameter(a)
    b_ = a_.expand(tuple(expand_shape))
    b_.backward()
    grad = np.full_like(a_.value, fill_value=n_copies)
    np.testing.assert_equal(a_.grad, grad)


def test_log() -> None:
    a = np.random.randn(3, 2) ** 2
    a_ = mpp.Parameter(a)
    c_ = a_.log()
    c_.backward()
    np.testing.assert_equal(a_.grad, 1.0 / a_.value)


def test_matmul() -> None:
    a = np.random.randn(4, 3)
    b = np.random.randn(3, 2)
    a_ = mpp.Parameter(a)
    b_ = mpp.Parameter(b)
    c_ = a_ @ b_
    c_.backward()
    for i, j in itertools.product(range(a.shape[0]), range(a.shape[1])):
        h = np.zeros_like(a)
        h[i, j] = 1e-6
        d = (((a + h) @ b - a @ b) / 1e-6).sum()
        assert pytest.approx(d) == a_.grad[i, j]
    for i, j in itertools.product(range(b.shape[0]), range(b.shape[1])):
        h = np.zeros_like(b)
        h[i, j] = 1e-6
        d = ((a @ (b + h) - a @ b) / 1e-6).sum()
        assert pytest.approx(d) == b_.grad[i, j]


@pytest.mark.parametrize("dim", DIMS)
def test_max(dim: int | tuple[int, ...] | None) -> None:
    a = np.random.randn(4, 3, 2)
    a_ = mpp.Parameter(a)
    b_ = a_.max(dim=dim)
    b_.backward()
    grad = (np.max(a, axis=dim, keepdims=True) == a).astype(a.dtype)
    np.testing.assert_equal(a_.grad, grad)


def test_maximum() -> None:
    a = np.random.randn(3, 2)
    b = np.random.randn(3, 2)
    a_ = mpp.Parameter(a)
    b_ = mpp.Parameter(b)
    c_ = mpp.maximum(a_, b_)
    c_.backward()
    grad = a > b
    np.testing.assert_equal(a_.grad, grad)
    np.testing.assert_equal(b_.grad, ~grad)


def test_mult() -> None:
    a = np.random.randn(4, 1, 2)
    b = np.random.randn(3, 2)
    a_ = mpp.Parameter(a)
    b_ = mpp.Parameter(b)
    c_ = a_ * b_
    c_.backward()
    a_grad = np.broadcast_to(b.sum(axis=0, keepdims=True), shape=(4, 1, 2))
    b_grad = np.broadcast_to(a.sum(axis=0), shape=(3, 2))
    np.testing.assert_equal(a_.grad, a_grad)
    np.testing.assert_equal(b_.grad, b_grad)


def test_mult_scalar() -> None:
    a = np.random.randn(3, 2)
    a_ = mpp.Parameter(a)
    c = 2.0
    b_ = c * a_
    b_.backward()
    grad = np.full_like(a, fill_value=c)
    np.testing.assert_equal(a_.grad, grad)


def test_no_grad() -> None:
    with mpp.no_grad():
        a = np.random.randn(4, 1, 2)
        b = np.random.randn(3, 2)
        a_ = mpp.Parameter(a)
        b_ = mpp.Parameter(b)
        c_ = a_ * b_
        with pytest.raises(ValueError):
            c_.backward()


def test_pow() -> None:
    a = np.random.randn(3, 2)
    a_ = mpp.Parameter(a)
    c_ = a_**3
    c_.backward()
    np.testing.assert_equal(a_.grad, 3 * a**2)


def test_relu() -> None:
    a = np.random.randn(3, 2)
    a_ = mpp.Parameter(a)
    b_ = mpp.relu(a_)
    b_.backward()
    np.testing.assert_equal(a_.grad, a > 0.0)


@pytest.mark.parametrize("dim", DIMS)
def test_squeeze(dim: int | tuple[int, ...] | None) -> None:
    shape = [4, 3, 2]
    if dim is None:
        dim = (0, 2)
    if isinstance(dim, int):
        dim = (dim,)
    for d in dim:
        shape[d] = 1
    a = np.random.randn(shape[0], shape[1], shape[2])
    a_ = mpp.Parameter(a)
    b_ = a_.squeeze(dim=dim)
    b_.backward()
    grad = np.ones_like(a)
    np.testing.assert_equal(a_.grad, grad)


@pytest.mark.parametrize("dim", DIMS)
def test_sum(dim: int | tuple[int, ...] | None) -> None:
    a = np.random.randn(4, 3, 2)
    a_ = mpp.Parameter(a)
    b_ = a_.sum(dim=dim)
    b_.backward()
    grad = np.ones_like(a)
    np.testing.assert_equal(a_.grad, grad)


def test_transpose() -> None:
    a = np.random.randn(4, 3)
    a_ = mpp.Parameter(a)
    b_ = a_.transpose(0, 1)
    np.testing.assert_equal(b_.value, a.T)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_unsqueeze(dim: int) -> None:
    a = np.random.randn(4, 3, 2)
    a_ = mpp.Parameter(a)
    b_ = a_.unsqueeze(dim)
    b_.backward()
    grad = np.ones_like(a)
    np.testing.assert_equal(a_.grad, grad)
