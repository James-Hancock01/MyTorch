from __future__ import annotations
import numpy as np
import pytest

from neuralnet.network import MLP, DenseLayer
from neuralnet.activators import ReLU, Sigmoid


def _make_model() -> MLP:
    return MLP(
        [
            DenseLayer(4, 8),
            ReLU(),
            DenseLayer(8, 8),
            Sigmoid(),
            DenseLayer(8, 3),
        ]
    )


def test_save_writes_expected_npz_structure(tmp_path):
    """
    Forces exact known parameter values (including a square 8x8 layer) and
    checks the raw .npz contents directly, bypassing MLP.load entirely.
    This isolates whether a bug lives in save() vs load()/set_parameters().
    """
    layer0 = DenseLayer(4, 8)
    layer0.set_parameters(
        weights=np.arange(4 * 8, dtype=float).reshape(4, 8),
        biases=np.arange(8, dtype=float),
    )
    layer1 = DenseLayer(8, 8)  # square — deliberately chosen
    layer1.set_parameters(
        weights=np.arange(8 * 8, dtype=float).reshape(8, 8),
        biases=np.arange(8, dtype=float) + 100,
    )
    layer2 = DenseLayer(8, 3)
    layer2.set_parameters(
        weights=np.arange(8 * 3, dtype=float).reshape(8, 3),
        biases=np.arange(3, dtype=float) + 200,
    )

    model = MLP([layer0, ReLU(), layer1, Sigmoid(), layer2])

    path = tmp_path / "model.npz"
    model.save(path)

    raw = np.load(path)

    assert list(raw["layer_kinds"]) == [
        "DenseLayer",
        "ReLU",
        "DenseLayer",
        "Sigmoid",
        "DenseLayer",
    ]

    np.testing.assert_array_equal(raw["weights_0"], layer0.weights)
    np.testing.assert_array_equal(raw["biases_0"], layer0.biases)
    np.testing.assert_array_equal(raw["weights_1"], layer1.weights)
    np.testing.assert_array_equal(raw["biases_1"], layer1.biases)
    np.testing.assert_array_equal(raw["weights_2"], layer2.weights)
    np.testing.assert_array_equal(raw["biases_2"], layer2.biases)


def test_save_load_params_match(tmp_path):
    model = _make_model()
    path = tmp_path / "model.npz"

    model.save(path)

    loaded = MLP.load(path)

    original_params = model.params
    loaded_params = loaded.params

    assert len(original_params) == len(loaded_params)
    for orig, load in zip(original_params, loaded_params):
        np.testing.assert_array_equal(orig["weights"], load["weights"])
        np.testing.assert_array_equal(orig["biases"], load["biases"])


def test_save_load_architecture_matches(tmp_path):
    model = _make_model()
    path = tmp_path / "model.npz"
    model.save(path)

    loaded = MLP.load(path)

    assert len(model.layers) == len(loaded.layers)
    for orig_layer, loaded_layer in zip(model.layers, loaded.layers):
        assert type(orig_layer) is type(loaded_layer)


def test_save_load_predictions_match(tmp_path):
    model = _make_model()
    path = tmp_path / "model.npz"
    model.save(str(path))

    loaded = MLP.load(str(path))

    rng = np.random.default_rng(0)
    x = rng.random((5, 4)).astype(float)

    original_out = model.forward(x)
    loaded_out = loaded.forward(x)

    np.testing.assert_array_almost_equal(original_out, loaded_out)


def test_load_rejects_unknown_activation(tmp_path):
    model = _make_model()
    path = tmp_path / "model.npz"
    model.save(str(path))

    # corrupt the saved layer_kinds to reference a nonexistent activation
    data = dict(np.load(path))
    data["layer_kinds"][1] = "TotallyFakeActivation"
    np.savez(path, **data)

    with pytest.raises(ValueError, match="Unknown activation type"):
        MLP.load(str(path))
