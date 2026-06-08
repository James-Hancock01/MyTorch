import pytest
import torch
import torch.nn as nn

from neuralnet.network import MLP
from neuralnet.activators import ReLU
from typing import List, Dict
from numpy import floating
from numpy.typing import NDArray

import numpy as np


def test_forward_propagation():
    torchmodel: nn.Sequential = nn.Sequential(
        nn.Linear(4, 8),  # nin = 4, nout = 8
        nn.ReLU(),
        nn.Linear(8, 2),  # nin = 8, nout = 2
        nn.Softmax(dim=1),
    )

    # Run a forward pass
    x = torch.randn(2, 4)  # (batch size, nin)
    torch_output = torchmodel(x)

    params: List[Dict[str, NDArray[floating]]] = []
    for module in torchmodel.modules():
        if isinstance(module, nn.Linear):
            params.append(
                {
                    "weights": module.weight.detach().numpy().T,
                    "biases": module.bias.detach().numpy(),
                }
            )

    model: MLP = MLP.from_parameters_array(params, gs=[ReLU()])

    my_output = model.forward(x.numpy())

    assert np.allclose(
        my_output, torch_output.detach().numpy()
    ), "Forward propagation invalid"


if __name__ == "__main__":
    test_forward_propagation()
