# MyTorch

A small homemade neural network library written using numpy only.

## Structure
```
neuralnet/
    network.py          MLP, DenseLayer
    activators.py       activation functions
    loss_functions.py   loss functions
    optimisers.py       optimisation routines
    data.py             routines for splitting datasets and iterating batches
    metrics.py          History (tracks and plots loss/accuracy)
tests/
examples/
```

## Install
pip install -r requirements.txt

## Quick start
```python
from neuralnet.network import MLP, DenseLayer
from neuralnet.activators import ReLU
from neuralnet.loss_functions import CategoricalCrossEntropy
from neuralnet.optimizers import SGDMomentum
from neuralnet.data import train_test_split, iterate_batches
from neuralnet.metrics import History

model = MLP([
    DenseLayer(784, 64), ReLU(),
    DenseLayer(64, 64), ReLU(),
    DenseLayer(64, 10),
])
loss_fn = CategoricalCrossEntropy()
optimizer = SGDMomentum(lr=0.01, momentum=0.9)

train_X, train_Y, val_X, val_Y = train_testt_split(X, y, ratio=0.1, seed=0)

for batch_X, batch_Y in iterate_batches(train_X, train_Y, batch_size=30):
    model.zero_grad()
    pred = model.forward(batch_X)
    loss = loss_fn.forward(pred, batch_Y)
    model.backward(loss_fn.backward())
    optimizer.step(model)
```

## Save / load a trained model
```python
model.save("model.npz")
load = MLP.load("model.npz")
```
## Tests
```bash
pytest tests/
```
