from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import deque
from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt


class Expr:
    """Represents a differentiable expression in the graph.

    Parameters
    ----------
    value
        Value
    children
        Sequence of subexpressions (those through which backpropagation occurs)
    label
        Human-readable name
    requires_grad
        If the gradient is not required, backpropagation will stop at this expression (if unspecified, the gradient is
        required if the gradient of at least one child is required)
    """

    def __init__(
        self,
        value: npt.NDArray,
        children: Sequence[Expr] = (),
        label: str | None = None,
        requires_grad: bool | None = None,
    ) -> None:
        self._value = value
        self._children = set(children)
        self._label = label
        if requires_grad is None:
            requires_grad = any(child._requires_grad for child in children)
        self._requires_grad = requires_grad
        self._grad = None

    def __repr__(self) -> str:
        d = {"value": self._value, "requires_grad": self._requires_grad}
        if self._label is not None:
            d["label"] = self._label
        args = ", ".join(f"{k}={v}" for k, v in d.items())
        return f"_Expr({args})"

    def __matmul__(self, other: Any) -> Expr:
        return _MatMul(self, other)

    def __add__(self, other: Any) -> Expr:
        if isinstance(other, int):
            other = float(other)
        if isinstance(other, float):
            return _AddScalar(self, other)
        return _Add(*_maybe_expand(self, other))

    def __getitem__(self, index: Any) -> Expr:
        return _Slice(self, index=index)

    def __truediv__(self, other: Any) -> Expr:
        return self * other ** (-1)

    def __mul__(self, other: Any) -> Expr:
        if isinstance(other, int):
            other = float(other)
        if isinstance(other, float):
            return _MultScalar(self, other)
        return _Mult(*_maybe_expand(self, other))

    def __neg__(self) -> Expr:
        return self * (-1.0)

    def __pow__(self, pow: Any) -> Expr:
        if not isinstance(pow, (int, float)):
            msg = f"Expected int or float exponent; received {pow}"
            raise ValueError(msg)
        return _Pow(self, pow)

    def __radd__(self, other: Any) -> Expr:
        return self + other

    def __rtruediv__(self, other: Any) -> Expr:
        return self / other

    def __rmatmul__(self, other: Any) -> Expr:
        return self @ other

    def __rmul__(self, other: Any) -> Expr:
        return self * other

    def __rsub__(self, other: Any) -> Expr:
        return (-self) + other

    def __sub__(self, other: Any) -> Expr:
        return self + (-other)

    def _backward(self, grad: npt.NDArray) -> None:
        del grad

    @lru_cache(maxsize=None)
    def _get_nodes(self) -> deque[Expr]:
        retval: deque[Expr] = deque()
        if not self._requires_grad:
            return retval
        visited: set[Expr] = set()
        marked: set[Expr] = set()

        def visit(node: Expr) -> None:
            if node in visited:
                return
            if node in marked:
                msg = "Detected cycle in gradient graph"
                raise RuntimeError(msg)
            marked.add(node)
            for child in node._children:
                if not child._requires_grad:
                    continue
                visit(child)
            marked.remove(node)
            visited.add(node)
            retval.appendleft(node)

        visit(self)
        return retval

    def backward(
        self,
        init: np.ndarray | float = 1.0,
        opt: Opt | None = None,
        retain_grad: bool | None = None,
    ):
        """Perform backpropagation and return all affected parameters.

        Suppose we call ``loss.backward()``.
        If ``loss`` is a scalar, then ``param.grad`` accumulates the derivative of ``loss`` with respect to ``param``.
        Otherwise, it accumulates the derivative of ``loss.sum()`` with respect to ``param``.
        If ``init`` is specified, it accumulates the derivative of ``(loss * init).sum()`` with respect to ``param``.

        Parameters
        ----------
        init
            Initial gradient
        opt
            If specified, the optimizer updates parameters using their respective gradients
        retain_grad
            Whether or not to deallocate gradients (if unspecified, gradients are deallocated if an optimizer is
            specified)
        """
        if not self._requires_grad:
            msg = "Attempted to perform backward pass on an expression that does not require a gradient"
            raise ValueError(msg)
        if retain_grad is None:
            retain_grad = opt is None
        self._grad = np.empty_like(self._value)
        self._grad[...] = init
        nodes = self._get_nodes()
        for node in nodes:
            assert node._grad is not None
            node._backward(node._grad)
            if opt is not None and len(node._children) == 0:
                opt.update_param(node)
            if not retain_grad:
                node._grad = None

    def exp(self) -> Expr:
        """Return the element-wise exponential."""
        return _Exp(self)

    def expand(self, shape: tuple[int, ...]) -> Expr:
        """Broadcast.

        Parameter
        ---------
        shape
            Shape to broadcast to
        """
        return _Expand(self, shape=shape)

    def log(self) -> Expr:
        """Take the element-wise natural logarithm."""
        return _Log(self)

    def max(self, dim: int | tuple[int, ...] | None = None) -> Expr:
        """Maximize across a dimension.

        Parameters
        ----------
        dim
            Axis or axes along which to operate. By default, all axes are used.
        """
        return _Max(a=self, dim=dim)

    def set_label(self, label: str) -> None:
        """Set the expression label."""
        self._label = label

    def squeeze(self, dim: int | tuple[int, ...] | None = None) -> Expr:
        """Remove axes of length one.

        Parameters
        ----------
        dim
            One or more axes to remove. By default, all length one axes are removed.
        """
        return _Squeeze(self, dim=dim)

    def sum(self, dim: int | tuple[int, ...] | None = None) -> Expr:
        """Sum across one or more dimensions.

        Parameters
        ----------
        dim
            Axis or axes along which to operate. By default, all axes are used.
        """
        return _Sum(self, dim=dim)

    def transpose(self, dim0: int, dim1: int) -> Expr:
        """Transpose axes.

        Parameters
        ----------
        dim0
            First dimension
        dim1
            Second dimension
        """
        return _Transpose(self, dim0=dim0, dim1=dim1)

    def update_grad(self, func: Callable[[], npt.NDArray]) -> None:
        """Update the gradient by adding to it the output of a function.

        Note that the function is only invoked if the gradient is required.
        """
        if not self._requires_grad:
            return
        if self._grad is None:
            self._grad = np.zeros_like(self._value)
        self._grad += func()

    def update_value(self, increment: npt.NDArray) -> None:
        """Update the value by adding to it."""
        self._value += increment

    def unsqueeze(self, dim: int) -> Expr:
        """Insert a dimension of size one at the specified position.

        Parameters
        ----------
        dim
            Position
        """
        return _Unsqueeze(self, dim=dim)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Data type."""
        return self._value.dtype

    @property
    def value(self) -> npt.NDArray:
        """Value."""
        return self._value.view()

    @property
    def grad(self) -> npt.NDArray:
        """Gradient."""
        if self._grad is None:
            msg = "Attempted to view untracked gradient"
            raise ValueError(msg)
        return self._grad.view()

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._value.ndim

    @property
    def requires_grad(self) -> bool:
        """Whether this expression requires its gradient be computed."""
        return self._requires_grad

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape."""
        return self._value.shape


class Opt(ABC):
    @abstractmethod
    def update_param(self, param: Expr) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass


class Constant(Expr):
    """An expression that does not require a gradient and has no children.

    Parameters
    ----------
    value
        Value
    label
        Human-readable name
    """

    def __init__(self, value: npt.NDArray, label: str | None = None) -> None:
        super().__init__(value=value, label=label)


class Parameter(Expr):
    """An expression that requires a gradient but has no children.

    Parameters
    ----------
    value
        Value
    label
        Human-readable name
    """

    def __init__(self, c: npt.NDArray, label: str | None = None) -> None:
        super().__init__(value=c, label=label, requires_grad=True)


def maximum(a: Expr, b: Expr) -> Expr:
    """The element-wise maximum of two expressions."""
    return _Maximum(*_maybe_expand(a, b))


def relu(expr: Expr) -> Expr:
    """The positive part of an expression."""
    return _ReLU(expr)


class _Add(Expr):
    def __init__(self, a: Expr, b: Expr) -> None:
        super().__init__(value=a._value + b._value, children=(a, b))
        self._a = a
        self._b = b

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad)
        self._b.update_grad(lambda: grad)


class _AddScalar(Expr):
    def __init__(self, a: Expr, b: float) -> None:
        super().__init__(value=a._value + b, children=(a,))
        self._a = a

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad)


class _Exp(Expr):
    def __init__(self, a: Expr) -> None:
        super().__init__(value=np.exp(a._value), children=(a,))
        self._a = a

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad * self._value)


class _Expand(Expr):
    def __init__(self, a: Expr, shape: tuple[int, ...]) -> None:
        # TODO(parsiad): Materializing a broadcast is expensive
        super().__init__(value=np.broadcast_to(a._value, shape=shape), children=(a,))
        self._a = a

    def _backward(self, grad: npt.NDArray) -> None:
        def func() -> npt.NDArray:
            dim = tuple(
                [
                    self.ndim - 1 - i
                    for i, (m, n) in enumerate(
                        itertools.zip_longest(
                            reversed(self._a.shape), reversed(self.shape)
                        )
                    )
                    if m is None or m != n
                ]
            )
            return grad.sum(axis=dim).reshape(self._a.shape)

        self._a.update_grad(func)


class _Log(Expr):
    def __init__(self, a: Expr) -> None:
        super().__init__(value=np.log(a._value), children=(a,))
        self._a = a

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad / self._a._value)


class _MatMul(Expr):
    def __init__(self, a: Expr, b: Expr) -> None:
        if not (a.ndim == 2 and b.ndim == 2):
            msg = "Matrix multiplication currently only supports 2-D inputs"
            raise NotImplementedError(msg)
        super().__init__(value=a._value @ b._value, children=(a, b))
        self._a = a
        self._b = b

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad @ self._b._value.T)
        self._b.update_grad(lambda: self._a._value.T @ grad)


class _Max(Expr):
    def __init__(self, a: Expr, dim: int | tuple[int, ...] | None) -> None:
        super().__init__(value=a._value.max(axis=dim, keepdims=True), children=(a,))
        self._a = a
        self._dim = dim

    def _backward(self, grad: npt.NDArray) -> None:
        def func() -> npt.NDArray:
            # TODO(parsiad): Materializing a mask is expensive
            mask = (self._a._value == self._value).astype(self._a._value.dtype)
            mask /= mask.sum(axis=self._dim, keepdims=True)
            return grad * mask

        self._a.update_grad(func)


class _Maximum(Expr):
    def __init__(self, a: Expr, b: Expr) -> None:
        super().__init__(value=np.maximum(a._value, b._value), children=(a, b))
        self._a = a
        self._b = b

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad * (self._a._value >= self._b._value))
        self._b.update_grad(lambda: grad * (self._a._value < self._b._value))


class _Mult(Expr):
    def __init__(self, a: Expr, b: Expr) -> None:
        super().__init__(value=a._value * b._value, children=(a, b))
        self._a = a
        self._b = b

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad * self._b._value)
        self._b.update_grad(lambda: self._a._value * grad)


class _MultScalar(Expr):
    def __init__(self, a: Expr, b: float) -> None:
        super().__init__(value=a._value * b, children=(a,))
        self._a = a
        self._b = b

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad * self._b)


class _Pow(Expr):
    def __init__(self, a: Expr, pow: int | float) -> None:
        super().__init__(value=a._value**pow, children=(a,))
        self._a = a
        self._pow = pow

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(
            lambda: grad * self._pow * self._a._value ** (self._pow - 1)
        )


class _ReLU(Expr):
    def __init__(self, a: Expr) -> None:
        super().__init__(value=np.maximum(a._value, 0.0), children=(a,))
        self._a = a

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad * (self._a._value > 0.0))


class _Slice(Expr):
    def __init__(self, a: Expr, index: Any) -> None:
        super().__init__(value=a._value[index], children=(a,))
        self._a = a
        self._index = index

    def _backward(self, grad: npt.NDArray) -> None:
        def func() -> npt.NDArray:
            backprop_grad = np.zeros_like(self._a._value)
            backprop_grad[self._index] = grad
            return backprop_grad

        self._a.update_grad(func)


class _Squeeze(Expr):
    def __init__(self, a: Expr, dim: int | tuple[int, ...] | None) -> None:
        super().__init__(value=a._value.squeeze(axis=dim), children=(a,))
        self._a = a

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad.reshape(self._a.shape))


class _Sum(Expr):
    def __init__(self, a: Expr, dim: int | tuple[int, ...] | None) -> None:
        super().__init__(value=a._value.sum(axis=dim, keepdims=True), children=(a,))
        self._a = a

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad)


class _Transpose(Expr):
    def __init__(self, a: Expr, dim0: int, dim1: int) -> None:
        self._axes = list(range(a.ndim))
        self._axes[dim0] = dim1
        self._axes[dim1] = dim0
        super().__init__(value=np.transpose(a._value, self._axes), children=(a,))
        self._a = a

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: np.transpose(grad, self._axes))


class _Unsqueeze(Expr):
    def __init__(self, a: Expr, dim: int) -> None:
        super().__init__(value=np.expand_dims(a._value, axis=dim), children=(a,))
        self._a = a
        self._dim = dim

    def _backward(self, grad: npt.NDArray) -> None:
        self._a.update_grad(lambda: grad.squeeze(axis=self._dim))


def _maybe_expand(a: Expr, b: Expr) -> tuple[Expr, Expr]:
    shape = np.broadcast_shapes(a.shape, b.shape)
    a_ = a if a.shape == shape else _Expand(a, shape=shape)
    b_ = b if b.shape == shape else _Expand(b, shape=shape)
    return a_, b_
