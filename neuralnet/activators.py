from __future__ import annotations
import numpy as np
from numpy import floating
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Any


class activation_function(ABC):
    """
    An activation function, used as a layer in the network,
    on the forward pass returns the value of the function
    on the backward pass returns the value of its derivative
    Caches its input on forward() so backward() can compute
    dL/dinput = dout * f'(input) without external help
    """

    def forward(
        self, Z: NDArray[floating], *args: Any, **kwargs: Any
    ) -> NDArray[floating]:
        self.x = Z  # cache for backward
        return self._apply(Z)

    def backward(
        self, dout: NDArray[floating], *args: Any, **kwargs: Any
    ) -> NDArray[floating]:
        return dout * self._derivative(self.x)

    @abstractmethod
    def _apply(self, Z: NDArray[floating]) -> NDArray[floating]: ...

    def _derivative(self, Z: NDArray[floating]) -> NDArray[floating]: ...


class ReLU(activation_function):
    """
    Rectified Linear Unit
    ReLU(Z) = Z if Z > 0 else 0

    dReLU(Z)/dZ = 1 if Z > 0 else 0
    """

    def _apply(self, Z: NDArray[floating]) -> NDArray[floating]:
        # np.maximum is element-wise so can use
        return np.maximum(0, Z)

    def _derivative(self, Z: NDArray[floating]) -> NDArray[floating]:
        return np.asarray(Z > 0, dtype=np.float64)


class Sigmoid(activation_function):
    """
    Sigmoid function
    sigmoid(Z) = 1 / ( 1 + exp(-Z) )

    dsigmoid(Z)/dZ = sigmoid(Z)(1-sigmoid(Z))
    """

    def _apply(self, Z: NDArray[floating]) -> NDArray[floating]:
        return 1 / (1 + np.exp(-Z))

    def _derivative(self, Z: NDArray[floating]) -> NDArray[floating]:
        return self._apply(Z) * (1 - self._apply(Z))


class Tanh(activation_function):
    """
    Tanh function
    tanh(Z) = ...

    dtanh(Z)/dZ = 1 / (1 - Z^2)
    """

    def _apply(self, Z: NDArray[floating]) -> NDArray[floating]:
        return np.tanh(Z)

    def _derivative(self, Z: NDArray[floating]) -> NDArray[floating]:
        return 1 - np.tanh(Z) ** 2


class Softmax(activation_function):
    """
    Stateless utility used by CategoricalCrossEntry, not a network layer.
    Kept separate from activation_function since it isn't used in MLP.layers
    """

    @classmethod
    def forward(cls, Z: NDArray[floating], dim: int = -1) -> NDArray[floating]:
        shifted = Z - np.max(Z, axis=dim, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=dim, keepdims=True)
