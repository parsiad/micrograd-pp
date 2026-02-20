import contextlib
import math
from collections.abc import Callable
from typing import Any, Generator

import numpy.typing as npt

from ._expr import Constant, Expr, Parameter, relu
from ._numpy import numpy as np
from ._func import cat, softmax
from ._util import n_samples


Module = Callable[[Expr], Expr]


_eval_mode = False


@contextlib.contextmanager
def eval() -> Generator[None, None, None]:
    """Context manager to switch to eval mode."""
    global _eval_mode
    state = _eval_mode
    _eval_mode = True
    yield
    _eval_mode = state


def is_eval() -> bool:
    """Determines whether or not eval mode is enabled."""
    return _eval_mode


def _broadcast_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return value


def _pad_2d(x: Expr, padding: tuple[int, int], fill_value: float) -> Expr:
    pad_h, pad_w = padding
    if pad_h == 0 and pad_w == 0:
        return x

    n, c, _, _ = x.shape

    if pad_w != 0:
        z = Constant(np.full((n, c, x.shape[2], pad_w), fill_value, dtype=x.dtype))
        x = cat((z, x, z), dim=3)

    if pad_h != 0:
        z = Constant(np.full((n, c, pad_h, x.shape[3]), fill_value, dtype=x.dtype))
        x = cat((z, x, z), dim=2)

    return x


def _validate_conv2d_parameters(
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> None:
    if min(*kernel_size) < 1:
        msg = "Kernel size must be positive"
        raise ValueError(msg)
    if min(*stride) < 1:
        msg = "Stride must be positive"
        raise ValueError(msg)
    if min(*padding) < 0:
        msg = "Padding must be nonnegative"
        raise ValueError(msg)
    if min(*dilation) < 1:
        msg = "Dilation must be positive"
        raise ValueError(msg)


class BatchNorm1d:
    """Batch normalization.

    Parameters
    ----------
    num_features
        Number of features
    affine
        Whether to use learnable scale and shift parameters
    dtype
        Data type for running mean and variance and scale and shift parameters
    eps
        When standardizing, this quantity is added to the denominator for numerical stability
    momentum
        Momentum used for the running mean and variance computations (if None, an ordinary average is computed)
    track_running_stats
        Whether to keep a running mean and variance
    label
        Human-readable name
    """

    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        dtype: type = np.float32,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        track_running_stats: bool = True,
        label: str | None = None,
    ) -> None:
        self._eps = eps
        self._momentum = momentum
        self._num_features = num_features
        if track_running_stats:
            self._running_mean = np.zeros((num_features,), dtype=dtype)
            self._running_var = np.ones((num_features,), dtype=dtype)
        else:
            self._running_mean = None
            self._running_var = None
        if affine:
            self._scale = Parameter(
                np.ones((num_features,), dtype=dtype), label=None if label is None else f"{label}/scale"
            )
            self._shift = Parameter(
                np.zeros((num_features,), dtype=dtype), label=None if label is None else f"{label}/shift"
            )
        else:
            self._scale = None
            self._shift = None
        self._n = 0

    def __call__(self, x: Expr) -> Expr:
        dim = (0,) + tuple(range(2, x.ndim))
        if self._running_mean is not None and self._running_var is not None and is_eval():
            mean = Constant(self._running_mean)
            var = Constant(self._running_var)
        else:
            mean = x.mean(dim=dim)
            var = x.var(dim=dim)
        if self._running_mean is not None and self._running_var is not None:
            increment = n_samples(dim, x.shape)
            n_new = self._n + increment
            if self._momentum is None:
                a = self._n / n_new
                b = increment / n_new
            else:
                a = 1.0 - self._momentum
                b = self._momentum
            self._running_mean = a * self._running_mean + b * mean.value
            self._running_var = a * self._running_var + b * var.value
            self._n = n_new
        shape = (1, x.shape[1]) + ((1,) * (x.ndim - 2))
        mean = mean.expand(shape)
        var = var.expand(shape)
        x_norm = (x - mean) / ((var + self._eps) ** 0.5)
        if self._scale is not None and self._shift is not None:
            return self._scale * x_norm + self._shift
        else:
            return x_norm

    def __repr__(self) -> str:
        return (
            f"BatchNorm1d({self._num_features}, eps={self._eps}, momentum={self._momentum}, "
            f"affine={self._scale is not None and self._shift is not None}, "
            f"track_running_stats={self._running_mean is not None and self._running_var is not None})"
        )


class Conv2d:
    """2D convolution over an input image composed of several input planes.

    Expects an input tensor of shape (N, C_in, H_in, W_in).
    Produces an output tensor of shape (N, C_out, H_out, W_out).

    Conv2D conventions are easiest to understand with an example.
    Consider a convolution with kernel size 4x3, dilation 2x5, and stride 6x7.
    The first patch has coordinates
    ```
    [(0,0), (0,5), (0,10),
     (2,0), (2,5), (2,10),
     (4,0), (4,5), (4,10),
     (6,0), (6,5), (6,10)]
    ```
    The second patch has coordinates
    ```
    [(0,7), (0,12), (0,17),
     (2,7), (2,12), (2,17),
     (4,7), (4,12), (4,17),
     (6,7), (6,12), (6,17)]
    ```

    Parameters
    ----------
    in_channels
        Number of channels in the input image
    out_channels
        Number of channels produced by the convolution
    kernel_size
        Size of the convolving kernel
    stride
        Controls how far the kernel moves across the image from one position to the next
    padding
        Controls the amount of zero-padding on both sides of the input
    dilation
        Controls how far apart the kernel elements are spaced from one position to the next within the kernel
    bias
        Whether or not to include a bias
    label
        Human-readable name
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        label: str | None = None,
    ) -> None:
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = _broadcast_pair(kernel_size)
        self._stride = _broadcast_pair(stride)
        self._padding = _broadcast_pair(padding)
        self._dilation = _broadcast_pair(dilation)
        _validate_conv2d_parameters(
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
        )

        std = 1.0 / math.sqrt(in_channels * math.prod(self._kernel_size))
        self._a = Parameter(
            std * np.random.randn(out_channels, in_channels, *self._kernel_size),
            label=None if label is None else f"{label}/weight",
        )
        if bias:
            self._b = Parameter(
                np.zeros((out_channels,)),
                label=None if label is None else f"{label}/bias",
            )
        else:
            self._b = None

    def __call__(self, x: Expr) -> Expr:
        x = _pad_2d(x, padding=self._padding, fill_value=0.0)
        n, _, h, w = x.shape

        kernel_h, kernel_w = self._kernel_size
        stride_h, stride_w = self._stride
        dilation_h, dilation_w = self._dilation

        support_h = dilation_h * (kernel_h - 1) + 1
        support_w = dilation_w * (kernel_w - 1) + 1

        out_h = 1 + (h - support_h) // stride_h
        out_w = 1 + (w - support_w) // stride_w

        w_flat = self._a.reshape((self._out_channels, -1)).transpose(0, 1)
        outputs = []
        for i in range(out_h):
            h_lo = i * stride_h
            h_hi = h_lo + support_h
            for j in range(out_w):
                w_lo = j * stride_w
                w_hi = w_lo + support_w
                patch = x[:, :, h_lo:h_hi:dilation_h, w_lo:w_hi:dilation_w]
                patch = patch.reshape((n, -1))
                y_ij = patch @ w_flat
                if self._b is not None:
                    y_ij = y_ij + self._b
                outputs.append(y_ij.unsqueeze(1))

        return cat(outputs, dim=1).reshape((n, out_h, out_w, self._out_channels)).transpose(1, 3).transpose(2, 3)

    def __repr__(self) -> str:
        return (
            f"Conv2d({self._in_channels}, {self._out_channels}, kernel_size={self._kernel_size}, "
            f"stride={self._stride}, padding={self._padding}, dilation={self._dilation}, "
            f"bias={self._b is not None})"
        )


class Dropout:
    """Dropout.

    Parameters
    ----------
    p
        Probability of an element to be zeroed
    gain
        Scaling multiplier used at test time (if unspecified, it is set to 1 / (1 - p))
    """

    def __init__(self, p: float, gain: float | None = None) -> None:
        if p < 0.0 or p > 1.0:
            msg = "Dropout probability has to be between zero and one"
            raise ValueError(msg)
        if gain is None:
            gain = 1.0 / (1.0 - p)
        self._gain = gain
        self._p = p

    def __call__(self, x: Expr) -> Expr:
        if is_eval():
            return x
        mask = Constant((np.random.random(size=x.shape) >= self._p).astype(x.dtype))
        return x * mask * self._gain


class Embedding:
    """Lookup table.

    Parameters
    ----------
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, label: str | None = None) -> None:
        self._a = Parameter(
            math.sqrt(3.0) * (2.0 * np.random.rand(num_embeddings, embedding_dim) - 1.0),
            label=label,
        )

    def __call__(self, x: npt.NDArray) -> Expr:
        return self._a[x]

    def __repr__(self) -> str:
        return f"Embedding({self._a.shape[0]}, {self._a.shape[1]})"


class LayerNorm:
    """Layer normalization.

    Parameters
    ----------
    normalized_shape
        Shape of the last D dimensions to normalize over where D is the dimension of normalized_shape (if an integer is
        specified, it will be promoted to a singleton tuple)
    bias
        Whether or not to learn a bias (ignored if elementwise_affine is False)
    dtype
        Data type for running mean and variance and scale and shift parameters
    elementwise_affine
        Whether to use learnable scale and shift parameters
    eps
        When standardizing, this quantity is added to the denominator for numerical stability
    label
        Human-readable name
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        bias: bool = True,
        dtype: type = np.float32,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        label: str | None = None,
    ) -> None:
        if not elementwise_affine and bias:
            msg = f"{LayerNorm.__name__} does not support learnable bias without a learnable scale"
            raise ValueError(msg)
        self._eps = eps
        self._scale = (
            Parameter(np.ones(normalized_shape, dtype=dtype), label=None if label is None else f"{label}/scale")
            if elementwise_affine
            else None
        )
        self._shift = (
            Parameter(np.zeros(normalized_shape, dtype=dtype), label=None if label is None else f"{label}/shift")
            if bias
            else None
        )
        self._normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape

    def __call__(self, x: Expr) -> Expr:
        dims = tuple(range(-len(self._normalized_shape), 0))
        mean = x.mean(dims, keepdim=True)
        var = x.var(dims, keepdim=True)
        retval = (x - mean) / ((var + self._eps) ** 0.5)
        if self._scale is not None:
            retval = self._scale * retval
        if self._shift is not None:
            retval = retval + self._shift
        return retval

    def __repr__(self) -> str:
        return (
            f"LayerNorm({self._normalized_shape}, eps={self._eps=}, "
            f"elementwise_affine={self._scale is not None}, "
            f"bias={self._shift is not None})"
        )


class Linear:
    """Linear layer.

    Parameters
    ----------
    in_features
        Number of input features
    out_features
        Number of output features
    bias
        Whether or not to include a bias
    label
        Human-readable name
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        label: str | None = None,
    ) -> None:
        self._a = Parameter(
            np.random.randn(out_features, in_features) / np.sqrt(in_features),
            label=None if label is None else f"{label}/weight",
        )
        if bias:
            self._b = Parameter(
                np.zeros((out_features,)),
                label=None if label is None else f"{label}/bias",
            )
        else:
            self._b = None

    def __call__(self, x: Expr) -> Expr:
        retval = x @ self._a.transpose(0, 1)
        if self._b is not None:
            retval = retval + self._b.expand(retval.shape)
        return retval

    def __repr__(self) -> str:
        return f"Linear({self._a.shape[1]}, {self._a.shape[0]})"


class MaxPool2d:
    """2D max pooling over an input image composed of several input planes.

    Expects an input tensor of shape (N, C, H_in, W_in).
    Produces an output tensor of shape (N, C, H_out, W_out).

    Parameters
    ----------
    kernel_size
        Size of the pooling window
    stride
        Controls how far the pooling window moves across the image from one position to the next
    padding
        Controls the amount of padding (negative infinity) on both sides of the input
    dilation
        Controls how far apart the sampled image elements are spaced from one position to the next within the window
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        self._kernel_size = _broadcast_pair(kernel_size)
        self._stride = self._kernel_size if stride is None else _broadcast_pair(stride)
        self._padding = _broadcast_pair(padding)
        self._dilation = _broadcast_pair(dilation)
        _validate_conv2d_parameters(
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
        )

    def __call__(self, x: Expr) -> Expr:
        x = _pad_2d(x, padding=self._padding, fill_value=-np.inf)
        n, c, h, w = x.shape

        kernel_h, kernel_w = self._kernel_size
        stride_h, stride_w = self._stride
        dilation_h, dilation_w = self._dilation

        support_h = dilation_h * (kernel_h - 1) + 1
        support_w = dilation_w * (kernel_w - 1) + 1

        out_h = 1 + (h - support_h) // stride_h
        out_w = 1 + (w - support_w) // stride_w

        outputs = []
        for i in range(out_h):
            h_lo = i * stride_h
            h_hi = h_lo + support_h
            for j in range(out_w):
                w_lo = j * stride_w
                w_hi = w_lo + support_w
                patch = x[:, :, h_lo:h_hi:dilation_h, w_lo:w_hi:dilation_w]
                y_ij = patch.max(dim=(2, 3))
                outputs.append(y_ij.unsqueeze(1))

        return cat(outputs, dim=1).reshape((n, out_h, out_w, c)).transpose(1, 3).transpose(2, 3)

    def __repr__(self) -> str:
        return (
            f"MaxPool2d(kernel_size={self._kernel_size}, stride={self._stride}, "
            f"padding={self._padding}, dilation={self._dilation})"
        )


class MultiheadAttention:  # TODO(parsiad): Finish docstring
    """Multi-Head Attention as described in [1].

    [1] Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N; Kaiser, Łukasz;
        Polosukhin, Illia (2017). "Attention is All you Need" (PDF). Advances in Neural Information Processing Systems.
        30. Curran Associates, Inc.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
    ) -> None:
        if not batch_first:
            msg = f"{MultiheadAttention.__name__} does not support batch dimension second; set batch_first=True"
            raise NotImplementedError(msg)

        if embed_dim % num_heads != 0:
            msg = "Embedding dimension must be divisible by number of heads"
            raise ValueError(msg)
        head_dim = embed_dim // num_heads
        self._num_heads = num_heads

        input_dq = embed_dim
        input_dk = embed_dim if kdim is None else kdim
        input_dv = embed_dim if vdim is None else vdim

        dk = head_dim
        dv = head_dim
        do = embed_dim

        self._wq = Linear(in_features=input_dq, out_features=num_heads * dk, bias=bias)
        self._wk = Linear(in_features=input_dk, out_features=num_heads * dk, bias=bias)
        self._wv = Linear(in_features=input_dv, out_features=num_heads * dv, bias=bias)
        self._wo = Linear(in_features=num_heads * dv, out_features=do, bias=bias)

        self._dropout = None if dropout == 0.0 else Dropout(dropout)

    def __call__(
        self,
        query: Expr,
        key: Expr,
        value: Expr,
        attn_mask: Expr | None = None,
        average_attn_weights: bool = True,
    ) -> tuple[Expr, Expr]:
        if attn_mask is not None and attn_mask.ndim != 2:
            msg = f"{MultiheadAttention.__name__} only supports two dimensional attention masks"
            raise NotImplementedError(msg)

        if not (query.ndim == key.ndim == value.ndim == 3):
            msg = "Expected three dimensional query, key, and value"
            raise ValueError(msg)

        batch_sz, seq_len, _ = query.shape

        q = query
        k = key
        v = value

        # Input projection
        q_wq = self._wq(q)  # (N, L, h * dk)
        k_wk = self._wk(k)  # (N, L, h * dk)
        v_wv = self._wv(v)  # (N, L, h * dv)

        # Introduce separate head dimension
        q_wq = q_wq.reshape((batch_sz, seq_len, self._num_heads, -1)).transpose(1, 2)  # (N, h, L, dk)
        k_wk = k_wk.reshape((batch_sz, seq_len, self._num_heads, -1)).transpose(1, 2)  # (N, h, L, dk)
        v_wv = v_wv.reshape((batch_sz, seq_len, self._num_heads, -1)).transpose(1, 2)  # (N, h, L, dv)

        # Compute each attention head
        logits = q_wq @ k_wk.transpose(2, 3) / math.sqrt(q_wq.shape[-1])  # (N, h, L, L)
        if attn_mask is not None:
            logits = logits + attn_mask
        attn_output_weights = softmax(logits, dim=3)
        if self._dropout is not None:
            attn_output_weights = self._dropout(attn_output_weights)
        heads = (attn_output_weights @ v_wv).transpose(1, 2)  # (N, L, h, dv)
        heads = heads.reshape((batch_sz, seq_len, -1))  # (N, L, h * dv)

        # Output projection
        attn_output = self._wo(heads)

        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        return attn_output, attn_output_weights


class ReLU:
    """Modular wrapper around the ReLU function."""

    def __call__(self, expr: Expr) -> Expr:
        return relu(expr)

    def __repr__(self) -> str:
        return "ReLU()"


class Sequential:
    """Sequential container of modules.

    Parameters
    ----------
    modules
        Zero or more modules
    """

    def __init__(self, *modules: Module) -> None:
        self._modules = modules

    def __call__(self, x: Expr) -> Expr:
        for module in self._modules:
            x = module(x)
        return x

    def __getitem__(self, index: Any) -> Expr:
        return self._modules[index]

    def __repr__(self) -> str:
        return f"Sequential({', '.join(str(module) for module in self._modules)})"
