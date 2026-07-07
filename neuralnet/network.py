from __future__ import annotations
from dataclasses import dataclass
from numpy import floating
from numpy.typing import NDArray
from neuralnet.activators import activation_function, Softmax, ReLU

import numpy as np
from typing import cast, List, Dict


@dataclass()
class DenseLayer:
    nin: int
    nout: int

    def __post_init__(
        self,
    ):
        self.weights = np.random.rand(self.nin, self.nout) - 0.5
        self.biases = np.random.rand(self.nout) - 0.5  # np.zeros((self.nout,))
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, x: NDArray[floating]) -> NDArray[floating]:
        """
        Calculate the forward propagated value for all neurons in the layer
        Applying the activation function g(Z)
        """
        self.x = x  # cache for backwards propagation
        return x @ self.weights + self.biases

    def backward(self, dout: NDArray[floating]) -> NDArray[floating]:
        """
        Calculate the dweights and dbiases from
        dout = dL/d(this layer's output)
        """
        self.dweights = self.x.T @ dout  # (nin, nout)
        self.dbiases = dout.sum(axis=0)  # (nout,)
        return dout @ self.weights.T  # dL/din - pass back to prev layer

    def zero_grad(self):
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    @property
    def params(self) -> Dict[str, NDArray[floating]]:
        """
        Return the network parameter list for the layer
        """
        return {
            "weights": self.weights,
            "biases": self.biases,
        }

    @property
    def grads(self) -> Dict[str, NDArray[floating]]:
        """
        Return the network gradient adjustments
        """
        return {
            "dweights": self.dweights,
            "dbiases": self.dbiases,
        }

    def set_parameters(
        self, weights: NDArray[floating], biases: NDArray[floating]
    ) -> None:
        weights_array = np.asarray(weights, dtype=float)
        biases_array = np.asarray(biases, dtype=float).reshape(-1)

        if weights_array.shape == (self.nout, self.nin):
            weights_array = weights_array.T
        elif weights_array.shape != (self.nin, self.nout):
            raise AssertionError(
                f"Expected weights shape {(self.nin, self.nout)} or {(self.nout, self.nin)}, got {weights_array.shape}"
            )

        if biases_array.shape != (self.nout,):
            raise AssertionError(
                f"Expected biases shape {(self.nout,)}, got {biases_array.shape}"
            )

        self.weights = weights_array
        self.biases = biases_array
        return


@dataclass
class MLP:
    layers: List[
        DenseLayer | activation_function
    ]  # alternating DenseLayer, ActivationFunction

    def __post_init__(self):
        pass
        # if not isinstance(self.layers[-1], Softmax):
        #     raise ValueError("Final layer should be softmax")

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

    def backward(self, loss_grad: NDArray[floating]):
        grad = np.asarray(loss_grad, dtype=float)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, lr: float):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.weights -= lr * layer.dweights  # apply gradients
                layer.biases -= lr * layer.dbiases

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.zero_grad()

    @property
    def params(self) -> List[Dict[str, NDArray[floating]]]:
        """
        Return an array of all layer parameter dictionaries
        """
        return [layer.params for layer in self.layers if isinstance(layer, DenseLayer)]

    @property
    def grads(self) -> List[Dict[str, NDArray[floating]]]:
        """
        Return an array of all layer parameter dictionaries
        """
        return [layer.grads for layer in self.layers if isinstance(layer, DenseLayer)]

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

        layers: List[DenseLayer | activation_function] = []
        nin: int = cast(int, parameters[0]["weights"].shape[0])

        for index, params in enumerate(parameters):
            weights = np.asarray(params["weights"], dtype=float)
            if weights.shape[0] == nin:
                nout = cast(int, weights.shape[1])
            elif weights.shape[1] == nin:
                nout = cast(int, weights.shape[0])
            else:
                raise AssertionError(
                    f"Cannot infer layer shape from weights with shape {weights.shape}"
                )

            layer = DenseLayer(nin, nout)
            layer.set_parameters(weights, params["biases"])
            layers.append(layer)

            if index < len(parameters) - 1:
                activation = gs[index] if index < len(gs) else ReLU()
                layers.append(activation)
            else:
                layers.append(Softmax())

            nin = nout

        mlp = cls(layers)
        return mlp
