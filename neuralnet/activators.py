from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from neuralnet.typing import Array
import numpy as np


class activation_function(ABC):
    """
    An activation function, used as a layer in the network,
    on the forward pass returns the value of the function
    on the backward pass returns the value of its derivative
    Caches its input on forward() so backward() can compute
    dL/dinput = dout * f'(input) without external help
    """

    def forward(self, Z: Array, *args: Any, **kwargs: Any) -> Array:
        self.x = Z  # cache for backward
        return self._apply(Z)

    def backward(self, dout: Array, *args: Any, **kwargs: Any) -> Array:
        return dout * self._derivative(self.x)

    @abstractmethod
    def _apply(self, Z: Array) -> Array: ...

    def _derivative(self, Z: Array) -> Array: ...


class ReLU(activation_function):
    """
    Rectified Linear Unit
    ReLU(Z) = Z if Z > 0 else 0

    dReLU(Z)/dZ = 1 if Z > 0 else 0
    """

    def _apply(self, Z: Array) -> Array:
        # np.maximum is element-wise so can use
        return np.maximum(0, Z)

    def _derivative(self, Z: Array) -> Array:
        return np.asarray(Z > 0, dtype=np.float64)


class Sigmoid(activation_function):
    """
    Sigmoid function
    sigmoid(Z) = 1 / ( 1 + exp(-Z) )

    dsigmoid(Z)/dZ = sigmoid(Z)(1-sigmoid(Z))
    """

    def _apply(self, Z: Array) -> Array:
        return 1 / (1 + np.exp(-Z))

    def _derivative(self, Z: Array) -> Array:
        return self._apply(Z) * (1 - self._apply(Z))


class Tanh(activation_function):
    """
    Tanh function
    tanh(Z) = ...

    dtanh(Z)/dZ = 1 / (1 - Z^2)
    """

    def _apply(self, Z: Array) -> Array:
        return np.tanh(Z)

    def _derivative(self, Z: Array) -> Array:
        return 1 - np.tanh(Z) ** 2


class Softmax(activation_function):
    """
    Stateless utility used by CategoricalCrossEntry, not a network layer.
    Kept separate from activation_function since it isn't used in MLP.layers
    """

    @classmethod
    def forward(cls, Z: Array, dim: int = -1) -> Array:
        shifted = Z - np.max(Z, axis=dim, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=dim, keepdims=True)
