from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from neuralnet.typing import Array
from neuralnet.activators import Softmax
import numpy as np


class loss_function(ABC):
    def forward(self, logits: Array, y_true: Array) -> Array: ...

    def backward(self) -> Array: ...


@dataclass
class CategoricalCrossEntropy(loss_function):

    def forward(self, logits: Array, y_true: Array) -> Array:
        self.probs = Softmax.forward(logits)
        self.y_true = y_true.astype(int)
        # loss per sample, then average over batch/population
        log_probs = -np.log(self.probs[np.arange(len(self.y_true)), self.y_true] + 1e-8)
        return log_probs.mean()

    def backward(self) -> Array:
        n = len(self.y_true)
        grad = self.probs.copy()
        grad[np.arange(n), self.y_true] -= 1  # subtract 1 from the correct class
        return grad / n  # average over batch
