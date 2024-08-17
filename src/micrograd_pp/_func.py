import numpy as np
import numpy.typing as npt

from ._expr import Expr


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
    input_max = input_.max(dim=1, keepdim=True)
    delta = input_ - input_max
    log_sum_exp = delta.exp().sum(dim=1).log().squeeze()
    return (log_sum_exp - delta[np.arange(n), target]).sum() / n
