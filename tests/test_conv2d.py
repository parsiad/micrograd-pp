from typing import Generator

import pytest

import micrograd_pp as mpp

np = mpp.numpy


@pytest.fixture(autouse=True)
def run_before_and_after_tests() -> Generator[None, None, None]:
    np.random.seed(0)
    yield


def _get_conv2d_reference(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> np.ndarray:
    import scipy.signal

    n, c_in, _, _ = x.shape
    c_out, _, kh, kw = w.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    support_h = dilation_h * (kh - 1) + 1
    support_w = dilation_w * (kw - 1) + 1
    out_h = 1 + (x_pad.shape[2] - support_h) // stride_h
    out_w = 1 + (x_pad.shape[3] - support_w) // stride_w

    y = np.zeros((n, c_out, out_h, out_w), dtype=x.dtype)

    for batch_idx in range(n):
        for out_channel in range(c_out):
            corr = np.zeros((x_pad.shape[2] - support_h + 1, x_pad.shape[3] - support_w + 1), dtype=x.dtype)
            for in_channel in range(c_in):
                dilated_kernel = np.zeros((support_h, support_w), dtype=w.dtype)
                dilated_kernel[::dilation_h, ::dilation_w] = w[out_channel, in_channel]
                corr += scipy.signal.correlate2d(
                    x_pad[batch_idx, in_channel],
                    dilated_kernel,
                    mode="valid",
                )
            if b is not None:
                corr += b[out_channel]
            y[batch_idx, out_channel] = corr[::stride_h, ::stride_w]
    return y


@pytest.mark.skipif(not pytest.importorskip("scipy.signal"), reason="Unable to import scipy.signal")
def test_conv2d_forward_matches_reference() -> None:
    x = np.random.randn(2, 3, 6, 7)
    w = np.random.randn(4, 3, 3, 2)
    b = np.random.randn(4)
    stride = (2, 1)
    padding = (1, 2)
    dilation = (1, 2)

    conv = mpp.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 2),
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )
    conv._a._value = w
    assert conv._b is not None
    conv._b._value = b

    y = conv(mpp.Constant(x))
    expected = _get_conv2d_reference(x=x, w=w, b=b, stride=stride, padding=padding, dilation=dilation)
    np.testing.assert_allclose(y.value, expected, atol=1e-12, rtol=0.0)
