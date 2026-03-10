"""MNIST data loader — download, cache, and serve as numpy arrays.

Downloads from the official MNIST mirror (ossci-datasets S3 bucket).
Caches gzipped files locally so subsequent runs are instant.

Adapted from the NumPy official tutorial:
https://numpy.org/numpy-tutorials/tutorial-deep-learning-on-mnist/
"""

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


@dataclass
class MNISTData:
    """MNIST dataset split into train, (optional) validation, and test sets.

    Attributes:
        train_images: Shape (N, 784), float64 in [0, 1].
        train_labels: Shape (N,), uint8 in [0, 9].
        test_images: Shape (10000, 784), float64 in [0, 1].
        test_labels: Shape (10000,), uint8 in [0, 9].
        val_images: Shape (M, 784) if split, else None.
        val_labels: Shape (M,) if split, else None.
    """

    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    val_images: np.ndarray | None = field(default=None)
    val_labels: np.ndarray | None = field(default=None)


def _download_file(url: str, dest: Path) -> None:
    """Download a file if it doesn't exist locally.

    Args:
        url: Full URL to download.
        dest: Local file path to save to.
    """
    import urllib.request

    if dest.exists():
        return

    print(f"Downloading {dest.name}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(dest, "wb") as f:
        f.write(response.read())


def load_mnist(data_dir: str | None = None) -> MNISTData:
    """Load the MNIST dataset, downloading if necessary.

    Args:
        data_dir: Directory to cache MNIST files. Defaults to
            benchmarks/.mnist_cache/ relative to this file.

    Returns:
        MNISTData with normalized images and integer labels.
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent / ".mnist_cache")

    cache = Path(data_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Download all files
    for fname in _FILES.values():
        _download_file(_BASE_URL + fname, cache / fname)

    # Load images: skip 16-byte header, reshape to (N, 784)
    def _load_images(key: str) -> np.ndarray:
        with gzip.open(cache / _FILES[key], "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float64) / 255.0

    # Load labels: skip 8-byte header
    def _load_labels(key: str) -> np.ndarray:
        with gzip.open(cache / _FILES[key], "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    return MNISTData(
        train_images=_load_images("train_images"),
        train_labels=_load_labels("train_labels"),
        test_images=_load_images("test_images"),
        test_labels=_load_labels("test_labels"),
    )


def split_validation(
    data: MNISTData, val_size: int = 10000, seed: int = 42
) -> MNISTData:
    """Split training data into train and validation sets.

    Uses a deterministic shuffle so results are reproducible.

    Args:
        data: MNIST data with full training set.
        val_size: Number of samples to hold out for validation.
        seed: Random seed for the shuffle.

    Returns:
        New MNISTData with reduced train set and populated val fields.
    """
    rng = np.random.default_rng(seed)
    n = data.train_images.shape[0]
    perm = rng.permutation(n)

    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    return MNISTData(
        train_images=data.train_images[train_idx],
        train_labels=data.train_labels[train_idx],
        test_images=data.test_images,
        test_labels=data.test_labels,
        val_images=data.train_images[val_idx],
        val_labels=data.train_labels[val_idx],
    )
