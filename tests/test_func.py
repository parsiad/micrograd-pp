import numpy as np
import scipy.special

import micrograd_pp as mpp


def test_softmax() -> None:
    a = np.random.randn(5, 4, 3)
    actual = mpp.softmax(mpp.Constant(a), dim=1).value
    desired = scipy.special.softmax(a, axis=1)
    np.testing.assert_allclose(actual, desired)
