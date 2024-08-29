import gzip
import hashlib
import os
import shutil
import sys
import urllib.request

import numpy.typing as npt

from .._numpy import numpy as np


def _compute_hash(file_path):
    with open(file_path, "rb") as file:
        md5_hash = hashlib.sha3_256()
        while chunk := file.read(8192):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def _load_array(name: str, hexdigest: str) -> npt.NDArray:
    if not (os.path.exists(f"/tmp/{name}.gz") and _compute_hash(f"/tmp/{name}.gz") == hexdigest):
        sys.stderr.write(f"downloading {name}...\n")
        urllib.request.urlretrieve(f"https://github.com/mkolod/MNIST/raw/master/{name}.gz", f"/tmp/{name}.gz")
    with gzip.open(f"/tmp/{name}.gz", "rb") as f_in:
        with open(f"/tmp/{name}", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return np.fromfile(f"/tmp/{name}", dtype="uint8")


def _load_images(name: str, hexdigest: str, normalize: bool) -> npt.NDArray:
    images = _load_array(name=name, hexdigest=hexdigest)[16:].astype(np.float32).reshape((-1, 28, 28))
    if normalize:
        images /= 255
        images -= 0.5
        images /= 0.5
    return images


def _load_labels(name: str, hexdigest: str) -> npt.NDArray:
    return _load_array(name, hexdigest=hexdigest)[8:]


def load_mnist(
    normalize: bool = True,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Load the normalized MNIST dataset.

    Parameters
    ----------
    normalize
        Whether to normalize the image values between -1 and 1.

    Returns
    -------
    train_images
        NumPy array of shape (n_train, 28, 28) where each entry is a float.
    train_labels
        NumPy array of shape (n_train, ) where each entry is an integer from 0 to 9 (inclusive) corresponding to a digit
    test_images
        NumPy array of shape (n_test, 28, 28) where each entry is a float.
    test_labels
        NumPy array of shape (n_test, ) where each entry is an integer from 0 to 9 (inclusive) corresponding to a digit
    """
    train_images = _load_images(
        name="train-images-idx3-ubyte",
        hexdigest="253303dac16f5399955fe0da9eb11b1a2fac8619e3d3607b56c16a81b3d4136d",
        normalize=normalize,
    )
    train_labels = _load_labels(
        name="train-labels-idx1-ubyte",
        hexdigest="db8d83a2b3a97185b5d716711c9668579f1e67de751d758e2f65860b92c1f382",
    )
    test_images = _load_images(
        name="t10k-images-idx3-ubyte",
        hexdigest="19440ef658014d7c82274a618108bcb97569304b11dbdf5f654351a7976538d4",
        normalize=normalize,
    )
    test_labels = _load_labels(
        name="t10k-labels-idx1-ubyte",
        hexdigest="a82ce3c45ec301a03143d29943560a30120afd142ae1144d60c29564fb549f48",
    )
    return train_images, train_labels, test_images, test_labels
