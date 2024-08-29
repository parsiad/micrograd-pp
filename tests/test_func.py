import pytest

import micrograd_pp as mpp

np = mpp.numpy


@pytest.mark.skipif(not pytest.importorskip("scipy.special"), reason="Unable to import scipy.special")
def test_softmax() -> None:
    import scipy.special

    a = np.random.randn(5, 4, 3)
    actual = mpp.softmax(mpp.Constant(a), dim=1).value
    desired = scipy.special.softmax(a, axis=1)
    np.testing.assert_allclose(actual, desired)
