import numpy as np

from neuralnet.activators import ReLU
from neuralnet.network import DenseLayer, MLP


def test_backward_propagation_matches_finite_differences():
    x = np.array([[1.0, -1.0]])
    target = np.array([[0.25]])

    model = MLP(
        [
            DenseLayer(2, 3),
            ReLU(),
            DenseLayer(3, 1),
        ]
    )
    if isinstance(model.layers[0], DenseLayer):
        model.layers[0].set_parameters(
            np.array([[0.2, -0.1, 0.3], [0.4, -0.2, 0.5]]),
            np.array([0.1, -0.2, 0.3]),
        )
    if isinstance(model.layers[2], DenseLayer):
        model.layers[2].set_parameters(
            np.array([[0.7], [0.8], [0.9]]),
            np.array([0.2]),
        )

    model.zero_grad()
    prediction = model.forward(x)
    loss_grad = 2.0 * (prediction - target)
    model.backward(loss_grad)

    eps = 1e-6

    for layer_idx, layer in enumerate(model.layers):
        if not isinstance(layer, DenseLayer):
            continue

        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                original = layer.weights[i, j]
                layer.weights[i, j] = original + eps
                loss_plus = np.sum((model.forward(x) - target) ** 2)

                layer.weights[i, j] = original - eps
                loss_minus = np.sum((model.forward(x) - target) ** 2)

                layer.weights[i, j] = original

                numerical_grad = (loss_plus - loss_minus) / (2 * eps)
                assert np.allclose(
                    layer.dweights[i, j], numerical_grad, atol=1e-5, rtol=1e-5
                ), f"weight gradient mismatch at layer {layer_idx}, ({i}, {j})"

        for i in range(layer.biases.shape[0]):
            original = layer.biases[i]
            layer.biases[i] = original + eps
            loss_plus = np.sum((model.forward(x) - target) ** 2)

            layer.biases[i] = original - eps
            loss_minus = np.sum((model.forward(x) - target) ** 2)

            layer.biases[i] = original

            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            assert np.allclose(
                layer.dbiases[i], numerical_grad, atol=1e-5, rtol=1e-5
            ), f"bias gradient mismatch at layer {layer_idx}, {i}"


if __name__ == "__main__":
    test_backward_propagation_matches_finite_differences()
