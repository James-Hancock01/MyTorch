import numpy as np

from neuralnet.network import DenseLayer, MLP


def test_model_update_applies_gradient_step_to_parameters():
    model = MLP([DenseLayer(2, 2), DenseLayer(2, 1)])

    first_layer = model.layers[0]
    if not isinstance(first_layer, DenseLayer):
        raise TypeError("first layer was not of type DenseLayer")
    first_layer.weights = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=float)
    first_layer.biases = np.array([0.1, -0.2], dtype=float)
    first_layer.dweights = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    first_layer.dbiases = np.array([0.05, -0.1])

    second_layer = model.layers[1]
    if not isinstance(second_layer, DenseLayer):
        raise TypeError("second layer was not of type DenseLayer")
    second_layer.weights = np.array([[0.7], [0.9]], dtype=float)
    second_layer.biases = np.array([0.25], dtype=float)
    second_layer.dweights = np.array([[0.2], [0.4]], dtype=float)
    second_layer.dbiases = np.array([0.3], dtype=float)

    model.update(lr=0.1)
    np.testing.assert_allclose(
        first_layer.weights,
        np.array([[0.99, -1.98], [0.47, 2.96]]),
    )
    np.testing.assert_allclose(first_layer.biases, np.array([0.095, -0.19]))
    np.testing.assert_allclose(
        second_layer.weights,
        np.array([[0.68], [0.86]]),
    )
    np.testing.assert_allclose(second_layer.biases, np.array([0.22]))
