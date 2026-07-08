from neuralnet.network import MLP, DenseLayer
from neuralnet.loss_functions import loss_function, CategoricalCrossEntropy
from neuralnet.activators import ReLU
from neuralnet.optimisers import SGDMomentum, OptimizerHealthError
from neuralnet.data import train_val_split, iterate_batches
from neuralnet.metrics import History
import numpy as np

trainset = np.load("trainset.npy")
full_train_X = trainset[:, :-1]
full_train_Y = trainset[:, -1]

testset = np.load("testset.npy")
test_X = testset[:, :-1]
test_Y = testset[:, -1]

full_train_X = full_train_X.astype("float32").reshape(-1, 28 * 28) / 255.0
test_X = test_X.astype("float32").reshape(-1, 28 * 28) / 255.0

# held-out validation split, separate from the final test set
train_X, train_Y, val_X, val_Y = train_val_split(
    full_train_X, full_train_Y, val_ratio=0.1, seed=0
)
print(f"{train_X.shape=}, {val_X.shape=}, {test_X.shape=}")

model = MLP(
    [
        DenseLayer(28**2, 64),
        ReLU(),
        DenseLayer(64, 64),
        ReLU(),
        DenseLayer(64, 10),
    ]
)
loss_fn = CategoricalCrossEntropy()
optimiser = SGDMomentum(lr=0.01, momentum=0.9)
history = History()

batch_size = 30
epochs = 50

for epoch in range(epochs):
    total_correct = 0
    total_loss = 0.0
    num_batches = 0

    for batch_X, batch_Y in iterate_batches(train_X, train_Y, batch_size, seed=epoch):
        model.zero_grad()
        pred = model.forward(batch_X)
        loss = loss_fn.forward(pred, batch_Y)
        dL = loss_fn.backward()
        model.backward(dL)

        try:
            optimiser.step(model)
        except OptimizerHealthError as e:
            print(f"Stopping early at epoch {epoch+1}: {e}")
            raise SystemExit(1)

        total_loss += loss
        total_correct += int((pred.argmax(axis=1) == batch_Y).sum())
        num_batches += 1

    train_loss = total_loss / num_batches
    train_acc = total_correct / len(train_X)
    val_loss, val_acc = evaluate(model, loss_fn, val_X, val_Y, batch_size=batch_size)

    history.record(train_loss, train_acc, val_loss, val_acc)
    print(
        f"Epoch {epoch+1}/{epochs}  "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.3%}  "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.3%}"
    )

history.plot(save_path="training_curves.png")

test_loss, test_acc = evaluate(model, loss_fn, test_X, test_Y)
print(f"Test loss {test_loss:.4f}, test accuracy {test_acc:.3%}")
