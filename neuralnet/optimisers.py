from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict
from neuralnet.typing import Array
from neuralnet.network import MLP, DenseLayer
import numpy as np


class Optimizer(ABC):
    """
    An optimizer updates a model's parameters in-place using
    the gradients already computed by MLP.backward().
    """

    @abstractmethod
    def step(self, model: MLP) -> None: ...


class OptimiserHealthError(RuntimeError):
    pass


@dataclass
class SGD(Optimizer):
    lr: float = 0.01

    def step(self, model: MLP) -> None:
        for layer in model.layers:
            if isinstance(layer, DenseLayer):
                layer.weights -= self.lr * layer.dweights
                layer.biases -= self.lr * layer.dbiases


@dataclass
class SGDMomentum(Optimizer):
    lr: float = 0.1
    momentum: float = 0.9
    grad_norm_eps: float = 1e-10  # below this, treat gradient as dead
    check_health: bool = True

    def __post_init__(self):
        # keyed by id(layer) so each DenseLayer instance gets its own velocity buffers
        self._velocities: Dict[int, Dict[str, Array]] = {}

    def step(self, model: MLP) -> None:
        total_grad_norm_sq = np.inf

        for layer in model.layers:
            if not isinstance(layer, DenseLayer):
                continue

            if self.check_health:
                if not (
                    np.isfinite(layer.dweights).all()
                    and np.isfinite(layer.dbiases).all()
                ):
                    raise OptimiserHealthError("NaN/Inf detected in gradients")
                total_grad_norm_sq += float((layer.dweights**2).sum())
                total_grad_norm_sq += float((layer.dbiases**2).sum())

            key = id(layer)
            if key not in self._velocities:
                self._velocities[key] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases),
                }
            v = self._velocities[key]

            v["weights"] = self.momentum * v["weights"] - self.lr * layer.dweights
            v["biases"] = self.momentum * v["biases"] - self.lr * layer.dbiases

            layer.weights += v["weights"]
            layer.biases += v["biases"]

        if total_grad_norm_sq < self.grad_norm_eps**2:
            raise OptimiserHealthError(
                f"Gradient norm collapsed to ~0 (norm^2={total_grad_norm_sq:.2e}) -"
                "training has stalled, likely dead-neurons or LR too high"
            )


@dataclass
class Adam(Optimizer):
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self):
        self._m: Dict[int, Dict[str, Array]] = {}
        self._v: Dict[int, Dict[str, Array]] = {}
        self._t: int = 0

    def step(self, model: MLP) -> None:
        self._t += 1
        for layer in model.layers:
            if not isinstance(layer, DenseLayer):
                continue

            key = id(layer)
            if key not in self._m:
                self._m[key] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases),
                }
                self._v[key] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases),
                }

            for name, dparam in (
                ("weights", layer.dweights),
                ("biases", layer.dbiases),
            ):
                m = self._m[key][name]
                v = self._v[key][name]

                m[:] = self.beta1 * m + (1 - self.beta1) * dparam
                v[:] = self.beta2 * v + (1 - self.beta2) * (dparam**2)

                m_hat = m / (1 - self.beta1**self._t)
                v_hat = v / (1 - self.beta2**self._t)

                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

                if name == "weights":
                    layer.weights -= update
                else:
                    layer.biases -= update
