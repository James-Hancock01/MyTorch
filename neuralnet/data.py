from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
from numpy import floating, integer
from numpy.typing import NDArray


def train_test_split(
    X: NDArray[floating],
    Y: NDArray[floating],
    ratio: float = 0.1,
    seed: int | None = None,
) -> Tuple[NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating]]:
    """
    Split (X, y) into (train_X, train_Y, test_X, test_Y).
    Shuffles before splitting so a pre-sorted dataset doesn't bias the split
    """
    assert 0.0 < ratio < 1.0, "ratio must be between 0 and 1"
    assert len(X) == len(Y), "X and Y must have the same number of examples"

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_test = int(len(X) * ratio)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def iterate_batches(
    X: NDArray[floating],
    Y: NDArray[floating],
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[Tuple[NDArray[floating], NDArray[floating]]]:
    """
    Yielld (batch_X, batch_Y) pairs covering the full dataset once
    """
    n = len(X)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], Y[batch_idx]
