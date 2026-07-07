from neuralnet.network import MLP, DenseLayer
from neuralnet.loss_functions import loss_function, CategoricalCrossEntropy
from neuralnet.activators import ReLU, Softmax
import numpy as np
from numpy.typing import NDArray
from typing import Any
from matplotlib import pyplot as plt

trainset = np.load("trainset.npy")
train_X = trainset[:, :-1]
train_Y = trainset[:, -1]
print(f"{train_X.shape=}, {train_Y.shape=}")

testset = np.load("testset.npy")
test_X = testset[:, :-1]
test_Y = testset[:, -1]
print(f"{test_X.shape=}, {test_Y.shape=}")


train_X = train_X.astype("float32").reshape(-1, 28 * 28) / 255.0
test_X = test_X.astype("float32").reshape(-1, 28 * 28) / 255.0


model = MLP(
    [
        DenseLayer(28**2, 10),
        ReLU(),
        DenseLayer(10, 10),
        ReLU(),
        DenseLayer(10, 10),
    ]
)
loss_fn = CategoricalCrossEntropy()


np.random.seed(0)

batch_size = 30
epochs = 500

for epoch in range(epochs):
    indices = np.random.permutation(len(train_X))
    total_correct = 0
    loss = 0

    for i in range(0, len(train_X), batch_size):
        batch_idx = indices[i : i + batch_size]
        batch_X = train_X[batch_idx]
        batch_Y = train_Y[batch_idx]

        model.zero_grad()
        pred = model.forward(batch_X)
        loss = loss_fn.forward(pred, batch_Y)
        dL = loss_fn.backward()

        model.backward(dL)
        model.update(lr=0.01)

        total_correct += int((pred.argmax(axis=1) == batch_Y).sum())

    accuracy = total_correct / len(train_X)
    if epoch % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, loss={loss:.4f}, accuracy {accuracy:.3%}")


def evaluate(
    model: MLP,
    loss_fn: loss_function,
    test_X: NDArray[np.floating[Any]],
    test_Y: NDArray[np.floating[Any]],
    batch_size: int = 32,
):
    total_loss = 0
    total_correct = 0
    num_batches = 0
    for i in range(0, len(test_X), batch_size):
        batch_X = test_X[i : i + batch_size]
        batch_Y = test_Y[i : i + batch_size]

        pred = model.forward(batch_X)
        loss = loss_fn.forward(pred, batch_Y)
        total_loss += loss
        total_correct += (pred.argmax(axis=1) == batch_Y).sum()
        num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = total_correct / len(test_X)

    print(f"Test loss {avg_loss:.4f}")
    print(f"Test accuracy {accuracy:.4f} ({total_correct}/{len(test_X)})")


evaluate(model, loss_fn, test_X, test_Y)
