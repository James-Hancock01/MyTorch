# MyTorch

A homemade neural network package to replicate PyTorch.

# 🧠 Neural Network Package — Build Roadmap

A checklist-driven roadmap for building a homemade neural network library in Python from scratch.

---

## Stage 1 — Project Scaffolding

> Set up a clean, professional project structure before writing any logic.

- [ ] Initialise a Git repository with a `.gitignore` for Python
- [ ] Create the package directory structure:
  ```
  neuralnet/
  ├── __init__.py
  ├── layers/
  ├── activations/
  ├── losses/
  ├── optimizers/
  ├── utils/
  └── model.py
  tests/
  examples/
  docs/
  ```
- [ ] Create `pyproject.toml` (or `setup.py`) with package metadata
- [ ] Add a `requirements.txt` (NumPy as the sole core dependency)
- [ ] Set up a virtual environment and confirm clean install
- [ ] Configure a test runner (`pytest`) and verify it discovers tests
- [ ] Add a `CHANGELOG.md` and this `ROADMAP.md`

**✅ Validation:** `python -c "import neuralnet"` runs without errors.

---

## Stage 2 — Core Math Primitives

> Build the low-level numerical foundation everything else depends on.

- [ ] Wrap NumPy arrays in a `Tensor` class (or use `np.ndarray` directly — decide and document the choice)
- [ ] Implement forward-pass matrix operations: dot product, addition, broadcasting
- [ ] Implement element-wise operations: multiply, divide, power, clip
- [ ] Write utility functions: `one_hot_encode`, `normalise`, `shuffle_dataset`
- [ ] Add a `seed` utility for reproducibility (`np.random.seed`)

**✅ Validation:** Unit-test every math primitive with known inputs and hand-calculated expected outputs.

---

## Stage 3 — Activation Functions

> Implement the non-linearities that give neural networks their expressive power.

- [ ] Create a base `Activation` class with `forward(x)` and `backward(x)` methods
- [ ] Implement **ReLU** — forward and derivative
- [ ] Implement **Sigmoid** — forward and derivative
- [ ] Implement **Tanh** — forward and derivative
- [ ] Implement **Softmax** — forward and Jacobian/simplified derivative
- [ ] Implement **Leaky ReLU** — with configurable `alpha`
- [ ] Implement **Linear** (identity) — for regression output layers
- [ ] Register activations in a lookup dict so they can be referenced by string name (e.g. `"relu"`)

**✅ Validation:**
- Plot each activation and its derivative over `[-5, 5]` and confirm shapes match reference graphs
- Assert `sigmoid(0) == 0.5`, `relu(-1) == 0`, `softmax` outputs sum to 1

---

## Stage 4 — Layer Implementations

> Build the composable building blocks of any network.

- [ ] Create a base `Layer` class with `forward()`, `backward()`, and `get_params()` interface
- [ ] Implement **`DenseLayer`** (fully connected):
  - [ ] Weight initialisation: He (ReLU), Xavier/Glorot (Sigmoid/Tanh)
  - [ ] Bias initialisation to zeros
  - [ ] Forward pass: `Z = XW + b`
  - [ ] Backward pass: compute `dW`, `db`, `dX`
  - [ ] Store `input`, `weights`, `biases` for backprop
- [ ] Implement **`DropoutLayer`**:
  - [ ] Apply inverted dropout mask during training
  - [ ] Pass through unchanged during inference
- [ ] Implement **`BatchNormLayer`**:
  - [ ] Forward pass with running mean/variance tracking
  - [ ] Backward pass for `gamma` and `beta` gradients
  - [ ] Separate `training` and `inference` modes

**✅ Validation:**
- Check `DenseLayer` output shape is `(batch_size, output_units)`
- Verify gradients numerically using finite differences (gradient check) on `DenseLayer`
- Confirm Dropout zeroes ~`rate`% of activations on average

---

## Stage 5 — Loss Functions

> Define what the network is optimising against.

- [ ] Create a base `Loss` class with `forward(y_pred, y_true)` and `backward()` methods
- [ ] Implement **Mean Squared Error (MSE)** — for regression
- [ ] Implement **Binary Cross-Entropy** — for binary classification
- [ ] Implement **Categorical Cross-Entropy** — for multi-class
- [ ] Implement **Softmax + Categorical Cross-Entropy** combined (numerically stable, faster backward pass)
- [ ] Add numerical stability guards (e.g. `np.clip` before `log`)

**✅ Validation:**
- Test each loss with perfect predictions → assert loss ≈ 0
- Test with worst-case predictions → assert loss is large and finite (no `inf` / `nan`)
- Gradient-check each loss backward against finite differences

---

## Stage 6 — Backpropagation Engine

> Wire the chain rule through the full network.

- [ ] Implement `Model.forward(X)` — passes input through all layers in sequence
- [ ] Implement `Model.backward(loss_grad)` — passes gradient backward through all layers in reverse
- [ ] Ensure each layer caches its inputs during forward pass for use in backward
- [ ] Accumulate `dW` and `db` for each `DenseLayer`
- [ ] Verify gradient flow doesn't vanish or explode on a small test network

**✅ Validation:**
- Run a full gradient check on a 2-layer network: numerical gradient vs. analytical gradient should agree to < `1e-5`
- Assert gradients are non-zero in all layers for a non-trivial input

---

## Stage 7 — Optimizers

> Implement the update rules that adjust weights using computed gradients.

- [ ] Create a base `Optimizer` class with an `update(layer)` method
- [ ] Implement **SGD** (Stochastic Gradient Descent) with learning rate
- [ ] Implement **SGD + Momentum**
- [ ] Implement **RMSProp**
- [ ] Implement **Adam** — with `beta1`, `beta2`, `epsilon`, and bias correction
- [ ] Add **learning rate decay** (step decay and exponential decay)
- [ ] Add **gradient clipping** (by value and by norm)

**✅ Validation:**
- Fit a single Dense layer to a linear function; confirm loss reaches near-zero within expected steps for each optimizer
- Verify Adam bias correction is applied on step 1 (check against reference implementation)

---

## Stage 8 — Model API

> Create a clean, user-facing interface to build and train networks.

- [ ] Implement `Model.add(layer)` for sequential layer stacking
- [ ] Implement `Model.compile(optimizer, loss)` to bind training configuration
- [ ] Implement `Model.fit(X, y, epochs, batch_size)`:
  - [ ] Mini-batch splitting and shuffling each epoch
  - [ ] Forward pass → loss → backward pass → optimizer step loop
  - [ ] Track and return loss (and optionally accuracy) per epoch
- [ ] Implement `Model.predict(X)` — forward pass in inference mode (no dropout)
- [ ] Implement `Model.evaluate(X, y)` — return loss and metrics on a dataset
- [ ] Add a `verbose` flag to print training progress

**✅ Validation:**
- Train on XOR problem (4 samples) and reach ~0 loss
- Train on small synthetic dataset and confirm `evaluate` loss matches manual calculation

---

## Stage 9 — Metrics & Evaluation

> Measure model performance beyond raw loss.

- [ ] Implement **Accuracy** (categorical and binary)
- [ ] Implement **Precision**, **Recall**, **F1-Score**
- [ ] Implement **Confusion Matrix**
- [ ] Implement **R² Score** for regression
- [ ] Add a `MetricsTracker` that logs values per epoch for plotting

**✅ Validation:**
- Test each metric with hand-crafted prediction/label pairs and assert exact expected values
- Confirm `accuracy([1,0,1], [1,0,1]) == 1.0` and `accuracy([1,0,1], [0,1,0]) == 0.0`

---

## Stage 10 — Regularisation

> Prevent overfitting with parameter penalties.

- [ ] Implement **L1 regularisation** — add `lambda * sum(|W|)` to loss; add sign gradient
- [ ] Implement **L2 regularisation** (weight decay) — add `lambda * sum(W²)` to loss; add `2 * lambda * W` to `dW`
- [ ] Integrate regularisation into `DenseLayer` and `Model.fit` loss computation
- [ ] Confirm Dropout (from Stage 4) integrates cleanly via the `training` flag

**✅ Validation:**
- Train an overfit-prone network without and with L2; assert val loss is lower with regularisation
- Check that L1 drives some weights close to zero

---

## Stage 11 — Serialisation (Save & Load)

> Persist trained models to disk.

- [ ] Implement `Model.save(filepath)` — serialise weights, biases, and architecture using `numpy.savez` or `pickle`
- [ ] Implement `Model.load(filepath)` — restore model to identical state
- [ ] Verify predictions are identical before saving and after loading

**✅ Validation:**
- Save a trained model, reload it, and assert `np.allclose(pred_before, pred_after)`

---

## Stage 12 — End-to-End Example Notebooks / Scripts

> Prove the library works on real tasks.

- [ ] **XOR** — 2-layer network, binary cross-entropy, ~100% accuracy
- [ ] **Iris classification** — 3-class softmax output, compare against `sklearn`
- [ ] **MNIST digit recognition** — demonstrate on a subset (1000 samples); target >90% accuracy
- [ ] **Sine wave regression** — MSE loss, linear output, low R² error
- [ ] Document each example with comments explaining every step

**✅ Validation:** All examples run end-to-end without errors and hit their stated accuracy/loss targets.

---

## Stage 13 — Documentation

> Make the package usable by others (and future you).

- [ ] Write a top-level `README.md` with: install instructions, quick-start code snippet, feature list
- [ ] Add NumPy-style docstrings to every public class and method
- [ ] Create a `docs/` folder with architecture decisions and math derivations
- [ ] Document each activation, loss, and optimizer with its formula
- [ ] Add type hints throughout the codebase

**✅ Validation:** A new user can install and run the quick-start example using only the README.

---

## Stage 14 — Testing & CI

> Lock in correctness and prevent regressions.

- [ ] Achieve >90% test coverage across all modules (`pytest-cov`)
- [ ] Add integration tests: full train/evaluate loop on XOR and Iris
- [ ] Add regression tests: save expected loss curves and assert new runs match within tolerance
- [ ] Set up GitHub Actions CI to run the test suite on every push
- [ ] Enforce a linter (`ruff` or `flake8`) and formatter (`black`) in CI

**✅ Validation:** CI pipeline passes green on a clean clone of the repository.

---

## Stage 15 — Optional Extensions

> Enhancements to tackle after the core is solid.

- [ ] **Convolutional layer** (`Conv2D`) — for image data
- [ ] **Recurrent layer** (`SimpleRNN` / `LSTM`) — for sequence data
- [ ] **Learning rate schedulers** — cosine annealing, warmup
- [ ] **Early stopping** callback
- [ ] **Data augmentation** utilities (for images)
- [ ] **GPU support** via CuPy (drop-in NumPy replacement)
- [ ] **Autograd engine** — replace manual `backward()` with automatic differentiation

---

## Progress Tracker

| Stage | Title | Status |
|-------|-------|--------|
| 1 | Project Scaffolding | ⬜ Not started |
| 2 | Core Math Primitives | ⬜ Not started |
| 3 | Activation Functions | ⬜ Not started |
| 4 | Layer Implementations | ⬜ Not started |
| 5 | Loss Functions | ⬜ Not started |
| 6 | Backpropagation Engine | ⬜ Not started |
| 7 | Optimizers | ⬜ Not started |
| 8 | Model API | ⬜ Not started |
| 9 | Metrics & Evaluation | ⬜ Not started |
| 10 | Regularisation | ⬜ Not started |
| 11 | Serialisation | ⬜ Not started |
| 12 | End-to-End Examples | ⬜ Not started |
| 13 | Documentation | ⬜ Not started |
| 14 | Testing & CI | ⬜ Not started |
| 15 | Optional Extensions | ⬜ Not started |
