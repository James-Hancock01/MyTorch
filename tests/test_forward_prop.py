import pytest
from neuralnet.network import MLP, DenseLayer
from neuralnet.activators import ReLU
from typing import List, Dict
from numpy import floating
from numpy.typing import NDArray

import numpy as np


def test_forward_propagation():
    x = np.array([1.0, -1.0])
    target = np.array([0.29])
    model = MLP(
        [
            DenseLayer(2, 3),
            ReLU(),
            DenseLayer(3, 1),
        ]
    )
    if not isinstance(model.layers[0], DenseLayer):
        raise TypeError()
    model.layers[0].set_parameters(
        np.array([[0.2, -0.1, 0.3], [0.4, -0.2, 0.5]]),
        np.array([0.1, -0.2, 0.3]),
    )

    if not isinstance(model.layers[2], DenseLayer):
        raise TypeError()
    model.layers[2].set_parameters(
        np.array([[0.7], [0.8], [0.9]]),
        np.array([0.2]),
    )
    model.zero_grad()
    prediction = model.forward(x)
    print(f"{prediction=}")
    assert np.isclose(prediction, target, atol=1e-5, rtol=1e-5)
    pass


if __name__ == "__main__":
    test_forward_propagation()
