from neuralnet.network import MLP
from neuralnet.activators import ReLU, Softmax
import numpy as np

"""
Define a Multi-Layer perceptron
MLP(inputs, ['n' layer sizes], ['n-1' activation functions])
"""
model = MLP(3, [10, 10, 2], [ReLU(), ReLU(), Softmax()])

"""
Define inputs in format
(population/batch size, sample size)
"""
X = np.asarray([[0.4, -0.4, 6.0], [0.4, -0.4, 6.0]], dtype=float)
print(X)
print(X.shape)
print(model.forward(X))


"""
Inspect the model parameters using...
"""
print("parameters", model.parameters)
