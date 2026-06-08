from __future__ import annotations
import numpy as np
from numpy import floating
from numpy.typing import NDArray
from abc import ABC
from typing import Any


class activation_function(ABC):
    """
    An activation function abstract base class,
    on the forward pass returns the value of the function
    on the backward pass returns the value of its derivative
    """

    @classmethod
    def forward(
        cls, Z: NDArray[floating], *args: Any, **kwargs: Any
    ) -> NDArray[floating]: ...

    @classmethod
    def backward(
        cls, Z: NDArray[floating], *args: Any, **kwargs: Any
    ) -> NDArray[floating]: ...


class ReLU(activation_function):
    """
    Rectified Linear Unit
    ReLU(Z) = Z if Z > 0 else 0

    dReLU(Z)/dZ = 1 if Z > 0 else 0
    """

    @classmethod
    def forward(cls, Z: NDArray[floating]) -> NDArray[floating]:
        # np.maximum is element-wise so can use
        return np.maximum(0, Z)

    @classmethod
    def backward(cls, Z: NDArray[floating]) -> NDArray[floating]:
        return np.asarray(Z > 0, dtype=np.float64)


class Sigmoid(activation_function):
    """
    Sigmoid function
    sigmoid(Z) = 1 / ( 1 + exp(-Z) )

    dsigmoid(Z)/dZ = sigmoid(Z)(1-sigmoid(Z))
    """

    @classmethod
    def forward(cls, Z: NDArray[floating]) -> NDArray[floating]:
        return 1 / (1 + np.exp(-Z))

    @classmethod
    def backward(cls, Z: NDArray[floating]) -> NDArray[floating]:
        return cls.forward(Z) * (1 - cls.forward(Z))


class Tanh(activation_function):
    """
    Tanh function
    tanh(Z) = ...

    dtanh(Z)/dZ = 1 / (1 - Z^2)
    """

    @classmethod
    def forward(cls, Z: NDArray[floating]) -> NDArray[floating]:
        return np.tanh(Z)

    @classmethod
    def backward(cls, Z: NDArray[floating]) -> NDArray[floating]:
        return 1 - np.tanh(Z) ** 2


class Softmax(activation_function):
    """
    Softmax function
    """

    @classmethod
    def forward(cls, Z: NDArray[floating], dim: int = -1) -> NDArray[floating]:
        shifted = Z - np.max(Z, axis=dim, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=dim, keepdims=True)

    @classmethod
    def backward(cls, Z: NDArray[floating]) -> NDArray[floating]:
        raise SyntaxError("Softmax function does not have a backwards")
