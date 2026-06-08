from __future__ import annotations
from dataclasses import dataclass
from re import X
from numpy import floating
from numpy.typing import NDArray
from neuralnet.activators import activation_function, Softmax

import numpy as np
from typing import cast, List, Dict


@dataclass()
class Neuron:
    nin: int
    # g: activation_function

    def __post_init__(
        self,
    ):
        self.weights = np.array([np.random.uniform(-1, 1) for _ in range(self.nin)])
        self.bias = np.random.uniform(-1, 1)

    def forward(self, x: NDArray[floating]) -> NDArray[floating]:
        """
        Calculate the forward propagated value from this neuron
        w . X + b
        Activation function applied at the layer level
        """
        self.x = x  # cache input
        # return x @ self.weights + self.bias
        return np.dot(self.weights, x) + self.bias

    def backward(self, dout: NDArray[floating]) -> NDArray[floating]:
        """
        Calculate the effect this neuron had on the final result
        where X is the array of effects the next layer had on the result.
        """
        self.dweights = dout * self.x  # dL/dw
        self.dbias = dout  # dL/db
        return dout * self.weights  # dL/din - pass back

    @property
    def parameters(self) -> NDArray[floating]:
        "Return a list of [weights, bias] for this neuron"
        return self.weights + np.array([self.bias])

    def set_parameters(self, weights: NDArray[floating], bias: floating) -> None:
        self.weights = weights
        self.bias = bias


@dataclass()
class Layer:
    nin: int
    nout: int
    g: activation_function

    def __post_init__(
        self,
    ):
        self.neurons = [Neuron(self.nin) for _ in range(self.nout)]
        sell

    def forward(self, X: NDArray[floating]) -> NDArray[floating]:
        """
        Calculate the forward propagated value for all neurons in the layer
        Applying the activation function g(Z)
        """
        Z = np.column_stack([n.forward(X) for n in self.neurons])
        return self.g.forward(Z)

    def backward(self, X: NDArray[floating]) -> NDArray[floating]:
        A = self.g.backward(X)
        Z = np.column_stack([n.backward(A) for n in self.neurons])
        return Z

    @property
    def parameters(self) -> NDArray[floating]:
        """
        Return the network parameter list [weights, bias] for each neuron in the layer
        """
        return np.array(
            [p for neuron in self.neurons for p in neuron.parameters], dtype=float
        )

    def set_parameters(
        self, weights: NDArray[floating], biases: NDArray[floating]
    ) -> None:
        assert weights.T.shape == (self.nout, self.nin)
        for n, ws, b in zip(self.neurons, weights.T, biases):
            print(f"{ws.shape=}, {b.shape=}")
            n.set_parameters(ws, b)


@dataclass
class MLP:
    nin: int
    nouts: List[int]
    gs: List[activation_function]

    def __post_init__(self):
        if len(self.gs) != len(self.nouts):
            raise ValueError(
                "Can only have as many activation functions as hidden layers"
            )

        if not isinstance(self.gs[-1], Softmax):
            raise ValueError("Final activation function should be softmax")

        self.sz: List[int] = [self.nin] + self.nouts
        self.layers = [
            Layer(nin=self.sz[i], nout=self.sz[i + 1], g=self.gs[i])
            for i in range(len(self.nouts))
        ]

    def forward(self, x: NDArray[floating]) -> NDArray[floating]:
        """
        Calculate the forward propagated values of the network
        """
        assert x.ndim <= 2, "X must be a 1D or 2D array"

        squeeze_output = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True

        for layer in self.layers:
            x = layer.forward(x)

        if squeeze_output:
            return x[0]

        return x

    @property
    def parameters(self) -> NDArray[floating]:
        """
        Return an array of all [weights, bias] for all neurons in all layers of the network
        """
        return np.array(
            [p for layer in self.layers for p in layer.parameters], dtype=float
        )

    @classmethod
    def from_parameters_array(
        cls,
        parameters: List[Dict[str, NDArray[floating]]],
        gs: List[activation_function],
    ) -> MLP:
        """
        Set the weights and biases of all neurons in all layers to predetermined values,
        useful for debugging.
        Takes parameters in the torch.nn.state_dict() format
        """

        # layers = [
        #     Layer(nin=p["weights"].shape[0], nout=p["weights"].shape[1], g=g)
        #     for p, g in zip(parameters, gs)
        # ]
        nin: int = cast(int, parameters[0]["weights"].shape[0])
        nouts: List[int] = [cast(int, p["weights"].shape[1]) for p in parameters]
        print("Network structure: ", [nin] + nouts)

        mlp = cls(nin=nin, nouts=nouts, gs=gs)

        for layer, p in zip(mlp.layers, parameters):
            layer.set_parameters(p["weights"], p["biases"])

        return mlp
