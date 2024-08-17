import contextlib
import math
from collections.abc import Callable
from typing import Any, Generator

import numpy as np
import numpy.typing as npt

from ._expr import Constant, Expr, Parameter, relu
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
    """

    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        dtype: type = np.float32,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        track_running_stats: bool = True,
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
            self._scale = Parameter(np.ones((num_features,), dtype=dtype))
            self._shift = Parameter(np.zeros((num_features,), dtype=dtype))
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
            f"BatchNorm1d({self._num_features}, x={self._eps=}, momentum={self._momentum}, "
            f"affine={self._scale is not None and self._shift is not None}, "
            f"track_running_stats={self._running_mean is not None and self._running_var is not None})"
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
        mask = Constant(np.random.random(size=x.shape) >= self._p)
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
