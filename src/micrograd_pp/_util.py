from ._numpy import numpy as np


def n_samples(dim: int | tuple[int, ...] | None, shape: tuple[int, ...]) -> int:
    if isinstance(dim, int):
        return shape[dim]
    if dim is None:
        return np.prod(np.array(shape)).item()
    return np.prod(np.array([shape[d] for d in dim])).item()
