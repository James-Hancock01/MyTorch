import pytest
from neuralnet.activators import ReLU, Sigmoid, Tanh, Softmax
from numpy import floating
from numpy.typing import NDArray

import numpy as np


def test_ReLU():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    y = ReLU.forward(x)
    assert np.all(
        y == np.array([0.0, 0.0, 1.0, 2.0])
    ), "Incorrect ReLU forward implementation"

    y = ReLU.backward(x)
    assert np.all(
        y == np.array([0.0, 0.0, 1.0, 1.0])
    ), "Incorrect ReLU backward implementation"


def test_Sigmoid():
    pass


if __name__ == "__main__":
    test_ReLU()
    test_Sigmoid()
