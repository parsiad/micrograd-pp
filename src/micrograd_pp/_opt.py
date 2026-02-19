from ._expr import Expr, Opt
from ._numpy import numpy as np


class SGD(Opt):
    """Performs stochastic gradient descent.

    Parameters
    ----------
    lr
        Learning rate
    """

    def __init__(self, lr: float) -> None:
        self._lr = lr

    def update_param(self, param: Expr) -> None:
        param.update_value(-self._lr * param.grad)

    def update_state(self) -> None:
        pass


class AdamW(Opt):
    """Performs Adam with decoupled weight decay (AdamW) optimization.

    The following diagram shows the historical evolution leading to AdamW:

        ┌───────> Momentum ────────┐
        |                          v
    SGD ┴> AdaGrad -> RMSProp -> Adam -> AdamW

    * Momentum smooths out gradient directions by an exponential moving average (EMA)
    * AdaGrad scales each coordinate step by how noisy it is over the entirety of training
    * RMSProp improves AdaGrad by using an EMA
    * Adam combines both Momentum and RMSProp
    * AdamW performs weight decay after the update step instead of modifying gradients directly
      (to avoid poisoning the Momentum and RMSProp-style statistics)

    Parameters
    ----------
    lr
        Learning rate
    betas
        Coefficients used for computing running averages of gradient and its square
    eps
        Term added to denominator for numerical stability
    weight_decay
        Decoupled weight decay coefficient (set to zero to recover Adam)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        beta_1, beta_2 = betas
        if lr <= 0.0:
            msg = "Learning rate must be positive"
            raise ValueError(msg)
        if eps <= 0.0:
            msg = "Epsilon must be positive"
            raise ValueError(msg)
        if not ((0.0 <= beta_1 < 1.0) and (0.0 <= beta_2 < 1.0)):
            msg = "Betas must be in the interval [0, 1)"
            raise ValueError(msg)
        if weight_decay < 0.0:
            msg = "Weight decay must be non-negative"
            raise ValueError(msg)

        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps
        self._weight_decay = weight_decay

        self._moments: dict[Expr, tuple[np.ndarray, np.ndarray]] = {}

        self._t = 0
        self.update_state()

    def update_param(self, param: Expr) -> None:
        if param not in self._moments:
            self._moments[param] = (np.zeros_like(param.grad), np.zeros_like(param.grad))

        moment_1, moment_2 = self._moments[param]
        moment_1[...] = self._beta_1 * moment_1 + (1.0 - self._beta_1) * param.grad
        moment_2[...] = self._beta_2 * moment_2 + (1.0 - self._beta_2) * (param.grad * param.grad)

        corrected_moment_1 = moment_1 / self._bias_correction_1
        corrected_moment_2 = moment_2 / self._bias_correction_2
        denom = np.sqrt(corrected_moment_2) + self._eps
        update = -self._lr * (corrected_moment_1 / denom + self._weight_decay * param.value)
        param.update_value(update)

    def update_state(self):
        self._t += 1
        self._bias_correction_1 = 1.0 - self._beta_1**self._t
        self._bias_correction_2 = 1.0 - self._beta_2**self._t
