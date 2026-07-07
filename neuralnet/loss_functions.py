from __future__ import annotations

from dataclasses import dataclass
from numpy import floating
from numpy.typing import NDArray

import numpy as np
from neuralnet.activators import Softmax
from abc import ABC


class loss_function(ABC):
    def forward(
        self, logits: NDArray[floating], y_true: NDArray[floating]
    ) -> NDArray[floating]: ...

    def backward(self) -> NDArray[floating]: ...


@dataclass
class CategoricalCrossEntropy(loss_function):

    def forward(
        self, logits: NDArray[floating], y_true: NDArray[floating]
    ) -> NDArray[floating]:
        self.probs = Softmax.forward(logits)
        self.y_true = y_true.astype(int)
        # loss per sample, then average over batch/population
        log_probs = -np.log(self.probs[np.arange(len(self.y_true)), self.y_true] + 1e-8)
        return log_probs.mean()

    def backward(self) -> NDArray[floating]:
        n = len(self.y_true)
        grad = self.probs.copy()
        grad[np.arange(n), self.y_true] -= 1  # subtract 1 from the correct class
        return grad / n  # average over batch
