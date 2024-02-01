from typing import Generator

import numpy as np
import pytest

import micrograd_pp as mpp

BATCH_SZ = 64
NUM_FEATURES = 10


@pytest.fixture(autouse=True)
def run_before_and_after_tests() -> Generator[None, None, None]:
    np.random.seed(0)
    yield


@pytest.mark.parametrize("momentum", [0.1, None])
def test_batch_norm_1d_track_running_stats(momentum: float) -> None:
    num_iters = 1_000
    shift = np.random.randn(10)
    scale = np.random.randn(10)
    bn = mpp.BatchNorm1d(NUM_FEATURES, affine=False, momentum=momentum)
    for _ in range(num_iters):
        x = scale * np.random.randn(BATCH_SZ, NUM_FEATURES) + shift
        x_ = mpp.Constant(x)
        bn(x_)
    assert bn._running_mean is not None
    assert bn._running_var is not None
    np.testing.assert_allclose(bn._running_mean, shift, atol=0.1, rtol=0.0)
    np.testing.assert_allclose(bn._running_var, scale * scale, atol=0.1, rtol=0.0)


def test_batch_norm_1d_standardize() -> None:
    shift = np.random.randn(10)
    scale = np.random.randn(10)
    bn = mpp.BatchNorm1d(NUM_FEATURES, affine=False)
    x = scale * np.random.randn(BATCH_SZ, NUM_FEATURES) + shift
    x_ = mpp.Constant(x)
    y_ = bn(x_)
    np.testing.assert_allclose(y_.value.mean(axis=0), 0.0, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(y_.value.var(axis=0), 1.0, atol=1e-3, rtol=0.0)


def test_batch_norm_1d_eval() -> None:
    shift = np.random.randn(10)
    scale = np.random.randn(10)
    bn = mpp.BatchNorm1d(NUM_FEATURES, affine=False)
    x = scale * np.random.randn(BATCH_SZ, NUM_FEATURES) + shift
    x_ = mpp.Constant(x)
    with mpp.eval():
        y_ = bn(x_)
        # The input should be close to the output since the batch norm scale and shift are 1 and 0 at initialization
        np.testing.assert_allclose(x_.value, y_.value, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("p", [-1.0, 2.0])
def test_dropout_bad_probabilities(p: float) -> None:
    with pytest.raises(ValueError):
        mpp.Dropout(p)


def test_dropout_eval() -> None:
    x = mpp.Constant(np.random.randn(BATCH_SZ, NUM_FEATURES))
    dropout = mpp.Dropout(0.5)
    with mpp.eval():
        y = dropout(x)
    np.testing.assert_equal(x.value, y.value)
