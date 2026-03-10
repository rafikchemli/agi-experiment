"""Forward-Forward algorithm — Hinton 2022.

Local learning without backpropagation. Each layer learns independently
by maximizing "goodness" (sum of squared activations) for positive data
and minimizing it for negative data.

Architecture: 784 -> 500 (ReLU) -> 500 (ReLU)
Learning: Per-layer Adam with local contrastive goodness loss
Expected accuracy: ~95-97% on MNIST (2-layer, 500-unit config)

Based on: "The Forward-Forward Algorithm: Some Preliminary Investigations"
Geoffrey Hinton, 2022. https://arxiv.org/abs/2212.13345

Section 3.3 — "A simple supervised example of FF":
- Labels encoded as one-hot in first 10 pixels of image
- Positive pass: image + correct label → high goodness
- Negative pass: image + wrong label → low goodness
- Inference: try all 10 labels, pick highest total goodness
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


def _overlay_labels(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Overlay one-hot label encoding onto the first 10 pixels.

    MNIST top-left pixels are nearly zero, so this is a clean embedding.
    The label value is scaled to the image's max pixel intensity.

    Args:
        images: Shape (N, 784), float64 in [0, 1].
        labels: Shape (N,), int in [0, 9].

    Returns:
        Modified images with one-hot labels in first 10 pixels.
    """
    result = images.copy()
    n = images.shape[0]
    result[:, :10] = 0.0
    max_vals = np.maximum(images.max(axis=1), 0.1)  # floor to avoid zero overlay
    result[np.arange(n), labels] = max_vals
    return result


def _random_wrong_labels(
    labels: np.ndarray, n_classes: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate random incorrect labels for negative samples.

    Args:
        labels: Correct labels, shape (N,).
        n_classes: Total number of classes.
        rng: Random generator.

    Returns:
        Wrong labels, shape (N,), guaranteed != correct labels.
    """
    wrong = rng.integers(0, n_classes - 1, size=labels.shape[0])
    # Shift values >= correct label up by 1 to avoid collision
    wrong[wrong >= labels] += 1
    return wrong


class _AdamState:
    """Per-parameter Adam optimizer state.

    Args:
        shape: Shape of the parameter tensor.
        lr: Learning rate.
        beta1: First moment decay.
        beta2: Second moment decay.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        lr: float = 0.03,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Apply one Adam update step.

        Args:
            param: Current parameter values.
            grad: Gradient of loss w.r.t. param.

        Returns:
            Updated parameter values.
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class _FFLayer:
    """A single Forward-Forward layer with local learning.

    Each layer independently learns to assign high goodness to positive
    inputs and low goodness to negative inputs. No gradient flows
    between layers.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        threshold: Goodness threshold separating positive from negative.
        lr: Adam learning rate.
        rng: Random generator for weight initialization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        threshold: float,
        lr: float,
        rng: np.random.Generator,
    ) -> None:
        self.threshold = threshold

        # Xavier initialization
        scale = np.sqrt(2.0 / input_dim)
        self.w = rng.normal(0, scale, (input_dim, output_dim))
        self.b = np.zeros(output_dim, dtype=np.float64)

        self._opt_w = _AdamState(self.w.shape, lr=lr)
        self._opt_b = _AdamState(self.b.shape, lr=lr)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: L2-normalize input, then linear + ReLU.

        Args:
            x: Input, shape (batch, input_dim).

        Returns:
            Activations, shape (batch, output_dim).
        """
        # L2 normalize input (key to FF — removes magnitude, keeps direction)
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return np.maximum(0, x_norm @ self.w + self.b)  # ReLU

    def goodness(self, activations: np.ndarray) -> np.ndarray:
        """Compute goodness as mean of squared activations per sample.

        Args:
            activations: Layer output, shape (batch, output_dim).

        Returns:
            Goodness scores, shape (batch,).
        """
        return np.mean(activations**2, axis=1)

    def train_step(
        self, x_pos: np.ndarray, x_neg: np.ndarray
    ) -> tuple[float, float, float]:
        """One training step with positive and negative inputs.

        Computes the local contrastive loss and updates weights via Adam.
        The loss pushes positive goodness above threshold and negative
        goodness below threshold.

        Args:
            x_pos: Positive inputs (correct labels), shape (batch, input_dim).
            x_neg: Negative inputs (wrong labels), shape (batch, input_dim).

        Returns:
            Tuple of (loss, mean_pos_goodness, mean_neg_goodness).
        """
        batch_n = x_pos.shape[0]
        theta = self.threshold

        # Forward pass — positive
        x_pos_norm = x_pos / (np.linalg.norm(x_pos, axis=1, keepdims=True) + 1e-8)
        z_pos = x_pos_norm @ self.w + self.b
        a_pos = np.maximum(0, z_pos)  # ReLU
        g_pos = np.mean(a_pos**2, axis=1)  # goodness per sample

        # Forward pass — negative
        x_neg_norm = x_neg / (np.linalg.norm(x_neg, axis=1, keepdims=True) + 1e-8)
        z_neg = x_neg_norm @ self.w + self.b
        a_neg = np.maximum(0, z_neg)
        g_neg = np.mean(a_neg**2, axis=1)

        # Loss: log(1 + exp(-g_pos + θ)) + log(1 + exp(g_neg - θ))
        # This is softplus applied to push g_pos > θ and g_neg < θ
        logit_pos = -g_pos + theta
        logit_neg = g_neg - theta
        loss_pos = np.log(1 + np.exp(np.clip(logit_pos, -20, 20)))
        loss_neg = np.log(1 + np.exp(np.clip(logit_neg, -20, 20)))
        loss = float(np.mean(loss_pos + loss_neg))

        # Gradients via chain rule (single layer, tractable)
        #
        # d(loss)/d(g_pos) = -sigmoid(-g_pos + θ)  (want to increase g_pos)
        # d(loss)/d(g_neg) = sigmoid(g_neg - θ)    (want to decrease g_neg)
        # d(g)/d(a) = 2*a/output_dim
        # d(a)/d(z) = (z > 0)  [ReLU derivative]
        # d(z)/d(w) = x_norm.T
        # d(z)/d(b) = 1
        #
        # We skip gradients through L2-norm (Hinton: "the weight updates
        # have no effect on the layer-normalized output for the same input")

        output_dim = self.w.shape[1]

        # Positive gradients
        dsig_pos = -1.0 / (1 + np.exp(-np.clip(logit_pos, -20, 20)))  # -sigmoid
        da_pos = (dsig_pos[:, None] * 2 * a_pos / output_dim) * (z_pos > 0).astype(
            np.float64
        )
        dw_pos = x_pos_norm.T @ da_pos / batch_n
        db_pos = np.mean(da_pos, axis=0)

        # Negative gradients
        dsig_neg = 1.0 / (1 + np.exp(-np.clip(logit_neg, -20, 20)))  # +sigmoid
        da_neg = (dsig_neg[:, None] * 2 * a_neg / output_dim) * (z_neg > 0).astype(
            np.float64
        )
        dw_neg = x_neg_norm.T @ da_neg / batch_n
        db_neg = np.mean(da_neg, axis=0)

        # Combined gradient
        dw = dw_pos + dw_neg
        db = db_pos + db_neg

        # Adam update
        self.w = self._opt_w.step(self.w, dw)
        self.b = self._opt_b.step(self.b, db)

        return loss, float(np.mean(g_pos)), float(np.mean(g_neg))


class ForwardForward(MNISTApproach):
    """Hinton's Forward-Forward algorithm for MNIST classification.

    Each layer trains independently with a local contrastive loss.
    No error signal propagates between layers — only activations flow
    forward. This makes FF biologically more plausible than backprop.

    Args:
        hidden_sizes: List of hidden layer sizes.
        threshold: Goodness threshold for the contrastive loss.
        learning_rate: Adam learning rate per layer.
        epochs: Training epochs per layer.
        batch_size: Mini-batch size.
        seed: Random seed for reproducibility.
        log_every: Print progress every N epochs.
    """

    name = "forward_forward"
    uses_backprop = False

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        threshold: float = 2.0,
        learning_rate: float = 0.03,
        epochs: int = 25,
        batch_size: int = 512,
        seed: int = 42,
        log_every: int = 5,
    ) -> None:
        self.hidden_sizes = hidden_sizes or [500, 500]
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.log_every = log_every

        self._layers: list[_FFLayer] = []

    def _quick_accuracy(
        self, images: np.ndarray, labels: np.ndarray, n_eval: int = 2000
    ) -> float:
        """Evaluate accuracy on a random subset for progress tracking.

        Args:
            images: Full image set.
            labels: Full label set.
            n_eval: Number of samples to evaluate on.

        Returns:
            Accuracy on the subset.
        """
        idx = self.rng.choice(len(images), size=min(n_eval, len(images)), replace=False)
        preds = self.predict(images[idx])
        return float(np.mean(preds == labels[idx]))

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train each layer greedily with the Forward-Forward algorithm.

        For each layer:
        1. Create positive samples (image + correct label overlay)
        2. Create negative samples (image + wrong label overlay)
        3. Train the layer to separate positive from negative by goodness
        4. Freeze the layer and pass data through it for the next layer

        Args:
            images: Training images, shape (N, 784), float64 in [0, 1].
            labels: Training labels, shape (N,), int in [0, 9].
        """
        n_samples = images.shape[0]

        # Build layers
        dims = [images.shape[1]] + self.hidden_sizes
        self._layers = []
        for i in range(len(self.hidden_sizes)):
            layer = _FFLayer(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                threshold=self.threshold,
                lr=self.learning_rate,
                rng=self.rng,
            )
            self._layers.append(layer)

        # Global epoch counter across all layers (for history tracking)
        global_epoch = 0

        for layer_idx, layer in enumerate(self._layers):
            print(f"    Layer {layer_idx+1}/{len(self._layers)} ({dims[layer_idx]}→{dims[layer_idx+1]})")

            for epoch in range(self.epochs):
                perm = self.rng.permutation(n_samples)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n_samples, self.batch_size):
                    idx = perm[start : start + self.batch_size]

                    # Create positive and negative samples at image level,
                    # then propagate through frozen preceding layers
                    x_pos = _overlay_labels(images[idx], labels[idx])
                    wrong = _random_wrong_labels(labels[idx], 10, self.rng)
                    x_neg = _overlay_labels(images[idx], wrong)

                    # Pass through all preceding frozen layers
                    for prev_layer in self._layers[:layer_idx]:
                        x_pos = prev_layer.forward(x_pos)
                        x_neg = prev_layer.forward(x_neg)

                    loss, g_pos, g_neg = layer.train_step(x_pos, x_neg)
                    epoch_loss += loss
                    n_batches += 1

                global_epoch += 1
                avg_loss = epoch_loss / n_batches

                # Evaluate accuracy and record history at log intervals
                if (epoch + 1) % self.log_every == 0 or epoch == 0:
                    train_acc = self._quick_accuracy(images, labels)
                    self.history.append(
                        EpochMetrics(
                            epoch=global_epoch,
                            train_acc=train_acc,
                            loss=avg_loss,
                        )
                    )
                    print(
                        f"      Epoch {epoch+1}/{self.epochs} — "
                        f"loss: {avg_loss:.4f}, g+: {g_pos:.2f}, g-: {g_neg:.2f}, "
                        f"acc: {train_acc:.4f}"
                    )

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict by trying all 10 labels and picking highest goodness.

        For each image, overlay each of 10 possible labels, forward through
        all layers, and sum the goodness across layers. The label producing
        the highest total goodness is the prediction.

        Args:
            images: Test images, shape (N, 784).

        Returns:
            Predicted labels, shape (N,), int in [0, 9].
        """
        if not self._layers:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        n = images.shape[0]
        goodness_per_label = np.zeros((n, 10), dtype=np.float64)

        for label in range(10):
            label_vec = np.full(n, label, dtype=np.intp)
            x = _overlay_labels(images, label_vec)

            total_goodness = np.zeros(n, dtype=np.float64)
            for layer in self._layers:
                x = layer.forward(x)
                total_goodness += layer.goodness(x)

            goodness_per_label[:, label] = total_goodness

        return np.argmax(goodness_per_label, axis=1).astype(np.uint8)

    def get_internals(self) -> dict[str, object]:
        """Expose layer weights and goodness statistics.

        Returns:
            Dict with weight matrices per layer.
        """
        internals: dict[str, object] = {}
        for i, layer in enumerate(self._layers):
            internals[f"w{i+1}"] = layer.w
            internals[f"b{i+1}"] = layer.b
        return internals
