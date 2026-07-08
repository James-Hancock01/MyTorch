from __future__ import annotations
from typing import cast, List, Dict
from pathlib import Path
from dataclasses import dataclass
from neuralnet.typing import Array
from neuralnet import activators
from neuralnet.activators import activation_function, Softmax, ReLU
import inspect
import numpy as np


def _build_activation_registry() -> Dict[str, type]:
    """
    Discover all activation_function subclasses in neuralnet.activators so
    MLP.save/load can identify them by class name
    """
    registry: Dict[str, type] = {}
    for name, obj in inspect.getmembers(activators, inspect.isclass):
        if (
            issubclass(obj, activation_function)
            and obj is not activation_function
            and not inspect.isabstract(obj)
        ):
            registry[name] = obj
    return registry


_ACTIVATION_REGISTRY = _build_activation_registry()


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

    def forward(self, x: Array) -> Array:
        """
        Calculate the forward propagated value for all neurons in the layer
        Applying the activation function g(Z)
        """
        self.x = x  # cache for backwards propagation
        return x @ self.weights + self.biases

    def backward(self, dout: Array) -> Array:
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
    def params(self) -> Dict[str, Array]:
        """
        Return the network parameter list for the layer
        """
        return {
            "weights": self.weights,
            "biases": self.biases,
        }

    @property
    def grads(self) -> Dict[str, Array]:
        """
        Return the network gradient adjustments
        """
        return {
            "dweights": self.dweights,
            "dbiases": self.dbiases,
        }

    def set_parameters(self, weights: Array, biases: Array) -> None:
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

    def forward(self, x: Array) -> Array:
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

    def backward(self, loss_grad: Array) -> Array:
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
    def params(self) -> List[Dict[str, Array]]:
        """
        Return an array of all layer parameter dictionaries
        """
        return [layer.params for layer in self.layers if isinstance(layer, DenseLayer)]

    @property
    def grads(self) -> List[Dict[str, Array]]:
        """
        Return an array of all layer parameter dictionaries
        """
        return [layer.grads for layer in self.layers if isinstance(layer, DenseLayer)]

    def save(self, path: Path) -> None:
        """
        Save weights, biases and enough architecture info to reconstruct this
        MLP later via MLP.load()
        """
        arrays: Dict[str, Array] = {}
        layer_kinds: List[str] = []

        dense_index = 0
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                arrays[f"weights_{dense_index}"] = layer.weights
                arrays[f"biases_{dense_index}"] = layer.biases
                layer_kinds.append(type(layer).__name__)
                dense_index += 1
            else:
                layer_kinds.append(type(layer).__name__)

        arrays["layer_kinds"] = np.array(layer_kinds, dtype="<U32")
        np.savez(path, **arrays, allow_pickle=True)

    @classmethod
    def load(cls, path: Path) -> MLP:
        """
        Reconstruct an MLP previously saved with MLP.save().
        """
        data = np.load(path)
        layer_kinds = data["layer_kinds"]

        layers: List[DenseLayer | activation_function] = []
        dense_index = 0

        for kind in layer_kinds:
            if kind == DenseLayer.__name__:
                weights = data[f"weights_{dense_index}"]
                biases = data[f"biases_{dense_index}"]
                nin, nout = weights.shape

                layer = DenseLayer(nin, nout)
                layer.weights = weights
                layer.biases = biases
                layers.append(layer)
                dense_index += 1
            else:
                if kind not in _ACTIVATION_REGISTRY:
                    raise ValueError(f"Unknown activation type '{kind}' in saved file")
                layers.append(_ACTIVATION_REGISTRY[kind]())

        return cls(layers)

    def update_parameters_from_dict(
        self,
        parameters: List[Dict[str, Array]],
        gs: List[activation_function],
    ) -> None:
        """
        Set the weights and biases of all neurons in all layers to predetermined values,
        useful for debugging.
        Takes parameters in the torch.nn.state_dict() format
        """

        dense_index = 0
        for layer in self.layers:
            if not isinstance(layer, DenseLayer):
                continue
            params = parameters[dense_index]
            weights = np.asarray(params["weights"], dtype=float)
            biases = np.asarray(params["biases"], dtype=float)
            assert (
                weights.shape == layer.weights.shape
            ), f"shape of weights does not match in layer {dense_index}"
            assert (
                biases.shape == layer.biases.shape
            ), f"shape of biases does not match in layer {dense_index}"

            layer.set_parameters(weights, biases)
            dense_index += 1
