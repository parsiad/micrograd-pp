from collections.abc import Callable
import numpy as np

from ._expr import Expr, Parameter, relu


Module = Callable[[Expr], Expr]


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
        return f"Linear({self._a.shape[0]}, {self._a.shape[1]})"


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

    def __repr__(self) -> str:
        return f"Sequential({', '.join(str(module) for module in self._modules)})"
