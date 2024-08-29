from typing import Sequence

import numpy.typing as npt

from ._numpy import numpy as np
from ._expr import Expr


class _Cat(Expr):
    def __init__(self, tensors: Sequence[Expr], dim: int) -> None:
        self._dim = dim
        self._tensors = tensors
        super().__init__(value=np.concatenate([tensor._value for tensor in tensors], axis=dim), children=tensors)

    def _backward(self, grad: npt.NDArray) -> None:
        lo = 0
        for tensor in self._tensors:
            hi = lo + tensor.shape[self._dim]
            tensor.update_grad(lambda lo_=lo, hi_=hi: grad.take(indices=np.arange(lo_, hi_), axis=self._dim))
            lo = hi


def _log_sum_exp(input_: Expr, dim: int) -> tuple[Expr, Expr]:
    input_max = input_.max(dim=dim, keepdim=True)
    delta = input_ - input_max
    log_sum_exp = delta.exp().sum(dim=dim, keepdim=True).log()
    return delta, log_sum_exp


def cat(tensors: Sequence[Expr], dim: int = 0) -> Expr:
    """Concatenates a sequence of tensors along a given dimension.

    Parameters
    ----------
    tensors
        Input tensors
    dim
        Dimension along which to concatenate

    Returns
    -------
    Concatenated tensor
    """
    return _Cat(tensors=tensors, dim=dim)


def cross_entropy_loss(input_: Expr, target: npt.NDArray) -> Expr:
    """Computes the cross entropy loss between input logits and target.

    Parameters
    ----------
    input_
        Logits
    target
        Class indices

    Returns
    -------
    Cross entropy loss
    """
    n, _ = input_.shape
    delta, log_sum_exp = _log_sum_exp(input_=input_, dim=1)
    return (log_sum_exp.squeeze() - delta[np.arange(n), target]).mean()


def softmax(input_: Expr, dim: int) -> Expr:
    """Computes the softmax.

    Parameters
    ----------
    input_
        Logits
    dim
        Dimension along which to compute softmax

    Returns
    -------
    Softmax
    """
    delta, log_sum_exp = _log_sum_exp(input_=input_, dim=dim)
    return (delta - log_sum_exp).exp()
