from ._expr import Expr, Opt


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

    def step(self) -> None:
        pass
