import math
from collections.abc import Iterable

import numpy.typing as npt

from ._expr import Expr
from ._numpy import numpy as np


def _get_grads(params: Iterable[Expr]) -> Iterable[npt.NDArray]:
    return (param.grad for param in params if param.requires_grad)


def clip_grad_value_(params: Iterable[Expr], clip_value: float) -> None:
    """Clip gradient values in-place.

    Parameters
    ----------
    params
        Parameters whose gradients should be clipped
    clip_value
        Maximum absolute gradient value
    """
    if clip_value < 0.0:
        msg = "clip_value must be non-negative"
        raise ValueError(msg)
    for grad in _get_grads(params):
        np.clip(grad, -clip_value, clip_value, out=grad)


def clip_grad_norm_(
    params: Iterable[Expr],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    eps: float = 1e-6,
) -> float:
    """Clip gradient norm in-place.

    Parameters
    ----------
    params
        Parameters whose gradients should be clipped
    max_norm
        Maximum allowed norm
    norm_type
        Type of p-norm to use. Supports ``math.inf`` for infinity norm.
    error_if_nonfinite
        If True, raises if the total norm is NaN or infinite
    eps
        Numerical stability term added to denominator
    """
    if max_norm < 0.0:
        msg = "max_norm must be non-negative"
        raise ValueError(msg)
    if eps <= 0.0:
        msg = "eps must be positive"
        raise ValueError(msg)
    if norm_type <= 0.0:
        msg = "norm_type must be positive"
        raise ValueError(msg)

    grads = list(_get_grads(params))
    if len(grads) == 0:
        return 0.0

    if math.isinf(norm_type):
        total_norm = max(float(np.abs(grad).max()) for grad in grads)
    else:
        total_norm = 0.0
        for grad in grads:
            total_norm += float((np.abs(grad) ** norm_type).sum())
        total_norm = total_norm ** (1.0 / norm_type)

    if error_if_nonfinite and not np.isfinite(total_norm):
        msg = f"The total norm of gradients is non-finite: {total_norm}"
        raise RuntimeError(msg)

    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for grad in grads:
            grad *= clip_coef

    return total_norm
