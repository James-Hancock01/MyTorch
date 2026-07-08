from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
from matplotlib import pyplot as plt

from neuralnet.data import iterate_batches
from neuralnet.network import MLP
from neuralnet.loss_functions import loss_function


@dataclass
class History:
    """
    Tracks per-epoch train/validation loss and accuracy, and can plot them.
    """

    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

    def record(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

    def plot(self, save_path: str | None = None) -> None:
        epochs = range(1, len(self.train_loss) + 1)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

        ax_loss.plot(epochs, self.train_loss, label="train")
        ax_loss.plot(epochs, self.val_loss, label="val")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()

        ax_acc.plot(epochs, self.train_acc, label="train")
        ax_acc.plot(epochs, self.val_acc, label="val")
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy")
        ax_acc.set_title("Accuracy")
        ax_acc.legend()

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        plt.show()


def evaluate(
    model: MLP,
    loss_fn: loss_function,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    batch_size: int = 32,
) -> Tuple[float, float]:
    """
    Returns (avg_loss, accuracy) over the given dataset.
    """
    total_loss = 0.0
    total_correct = 0
    num_batches = 0

    for batch_X, batch_y in iterate_batches(X, y, batch_size, shuffle=False):
        pred = model.forward(batch_X)
        loss = loss_fn.forward(pred, batch_y)
        total_loss += loss
        total_correct += int((pred.argmax(axis=1) == batch_y).sum())
        num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = total_correct / len(X)
    return avg_loss, accuracy
