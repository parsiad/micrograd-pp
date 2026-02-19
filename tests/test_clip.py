import pytest

import micrograd_pp as mpp

np = mpp.numpy


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    np.random.seed(0)
    yield


def _set_grad(param: mpp.Expr, grad: np.ndarray) -> None:
    param.zero_grad()
    param.update_grad(lambda: grad)


def test_clip_grad_value_clamps_each_element() -> None:
    param = mpp.Parameter(np.array([0.0, 0.0, 0.0]))
    _set_grad(param, np.array([-2.0, 0.25, 3.0]))

    mpp.clip_grad_value_([param], clip_value=0.5)

    np.testing.assert_allclose(param.grad, np.array([-0.5, 0.25, 0.5]))


def test_clip_grad_norm_scales_all_grads_by_common_factor() -> None:
    p1 = mpp.Parameter(np.zeros((2,)))
    p2 = mpp.Parameter(np.zeros((1,)))
    _set_grad(p1, np.array([3.0, 4.0]))
    _set_grad(p2, np.array([12.0]))

    total_norm = mpp.clip_grad_norm_([p1, p2], max_norm=6.5, norm_type=2.0)
    scale = 6.5 / (13.0 + 1e-6)

    np.testing.assert_allclose(total_norm, 13.0)
    np.testing.assert_allclose(p1.grad, np.array([3.0, 4.0]) * scale, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(p2.grad, np.array([12.0]) * scale, atol=1e-12, rtol=0.0)


def test_clip_grad_norm_noop_when_within_threshold() -> None:
    p1 = mpp.Parameter(np.zeros((2,)))
    p2 = mpp.Parameter(np.zeros((1,)))
    _set_grad(p1, np.array([3.0, 4.0]))
    _set_grad(p2, np.array([12.0]))

    total_norm = mpp.clip_grad_norm_([p1, p2], max_norm=13.1, norm_type=2.0)

    np.testing.assert_allclose(total_norm, 13.0)
    np.testing.assert_allclose(p1.grad, np.array([3.0, 4.0]), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(p2.grad, np.array([12.0]), atol=1e-12, rtol=0.0)


def test_clip_grad_norm_errors_on_nonfinite_if_requested() -> None:
    p = mpp.Parameter(np.zeros((1,)))
    _set_grad(p, np.array([np.inf]))

    with pytest.raises(RuntimeError):
        mpp.clip_grad_norm_([p], max_norm=1.0, error_if_nonfinite=True)
