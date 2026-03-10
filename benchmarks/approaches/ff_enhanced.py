"""Enhanced Forward-Forward — wider layers, LR scheduling, multi-negative training.

Key improvements over base Forward-Forward (Hinton 2022):
1. 2×1000 hidden layers (2× wider — more capacity for goodness separation)
2. Cosine LR schedule with warmup (better convergence in 25 epochs per layer)
3. AdamW-style decoupled weight decay (regularization)
4. Multiple negative samples per positive (richer gradient signal)
5. He initialization (better suited for ReLU than Xavier)
6. Larger batch size (2048 — smoother gradients with higher base LR)

Architecture: 784 → 1000 (L2-norm + ReLU) → 1000 (L2-norm + ReLU)

Biological analogues (beyond base FF):
- Wider layers ≈ cortical columns with more neurons (V1 has ~200M neurons)
- LR decay ≈ metaplasticity / synaptic consolidation (BCM theory)
- Multi-negative ≈ predictive coding with multiple competing hypotheses
- Weight decay ≈ synaptic turnover / protein degradation

Based on: "The Forward-Forward Algorithm: Some Preliminary Investigations"
Geoffrey Hinton, 2022. https://arxiv.org/abs/2212.13345
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


def _overlay_labels(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Overlay one-hot label encoding onto the first 10 pixels.

    Args:
        images: Shape (N, 784), float64 in [0, 1].
        labels: Shape (N,), int in [0, 9].

    Returns:
        Modified images with one-hot labels in first 10 pixels.
    """
    result = images.copy()
    n = images.shape[0]
    result[:, :10] = 0.0
    max_vals = np.maximum(images.max(axis=1), 0.1)
    result[np.arange(n), labels] = max_vals
    return result


def _random_wrong_labels(
    labels: np.ndarray, n_classes: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate random incorrect labels.

    Args:
        labels: Correct labels, shape (N,).
        n_classes: Total number of classes.
        rng: Random generator.

    Returns:
        Wrong labels, shape (N,), guaranteed != correct labels.
    """
    wrong = rng.integers(0, n_classes - 1, size=labels.shape[0])
    wrong[wrong >= labels] += 1
    return wrong


class _CosineAdam:
    """Adam optimizer with cosine LR schedule, warmup, and weight decay.

    Implements AdamW (decoupled weight decay) with cosine annealing
    learning rate schedule and optional linear warmup.

    Biological analogue: Metaplasticity — synaptic learning rates decrease
    as synapses consolidate, analogous to the BCM sliding threshold.

    Args:
        shape: Parameter tensor shape.
        lr_max: Peak learning rate (after warmup).
        lr_min: Minimum learning rate (at end of schedule).
        total_steps: Total optimization steps for the cosine schedule.
        warmup_steps: Linear warmup steps from 0 to lr_max.
        weight_decay: Decoupled weight decay coefficient.
        beta1: First moment exponential decay rate.
        beta2: Second moment exponential decay rate.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        lr_max: float = 0.06,
        lr_min: float = 0.001,
        total_steps: int = 750,
        warmup_steps: int = 90,
        weight_decay: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def _lr(self) -> float:
        """Current learning rate from cosine schedule with warmup."""
        if self.t <= self.warmup_steps:
            return self.lr_max * self.t / max(self.warmup_steps, 1)
        progress = (self.t - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * progress)
        )

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """One AdamW update step with scheduled learning rate.

        Args:
            param: Current parameter values.
            grad: Gradient of loss w.r.t. param.

        Returns:
            Updated parameter values.
        """
        self.t += 1
        lr = self._lr()

        # Decoupled weight decay (AdamW — not in gradient, applied directly)
        param = param * (1 - lr * self.weight_decay)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return param - lr * m_hat / (np.sqrt(v_hat) + self.eps)


class _FFLayerWide:
    """Forward-Forward layer with wider architecture and scheduled optimizer.

    Uses He initialization (sqrt(2/fan_in)) for ReLU activations and
    CosineAdam with warmup/decay for better convergence in limited epochs.
    Supports multiple negative samples per training step for richer gradients.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension (1000 recommended).
        threshold: Goodness threshold separating positive from negative.
        lr_max: Peak Adam learning rate.
        lr_min: Minimum Adam learning rate.
        total_steps: Total optimization steps for LR schedule.
        warmup_steps: Linear warmup steps.
        weight_decay: Decoupled weight decay coefficient.
        rng: Random generator for initialization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        threshold: float,
        lr_max: float,
        lr_min: float,
        total_steps: int,
        warmup_steps: int,
        weight_decay: float,
        rng: np.random.Generator,
    ) -> None:
        self.threshold = threshold

        # He initialization — optimal for ReLU (Kaiming et al. 2015)
        scale = np.sqrt(2.0 / input_dim)
        self.w = rng.normal(0, scale, (input_dim, output_dim))
        self.b = np.zeros(output_dim, dtype=np.float64)

        self._opt_w = _CosineAdam(
            self.w.shape, lr_max, lr_min, total_steps, warmup_steps, weight_decay
        )
        # No weight decay on bias (standard practice)
        self._opt_b = _CosineAdam(
            self.b.shape, lr_max, lr_min, total_steps, warmup_steps, 0.0
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: L2-normalize input, then linear + ReLU.

        Args:
            x: Input, shape (batch, input_dim).

        Returns:
            Activations, shape (batch, output_dim).
        """
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return np.maximum(0, x_norm @ self.w + self.b)

    def goodness(self, activations: np.ndarray) -> np.ndarray:
        """Compute goodness as mean of squared activations per sample.

        Args:
            activations: Layer output, shape (batch, output_dim).

        Returns:
            Goodness scores, shape (batch,).
        """
        return np.mean(activations**2, axis=1)

    def train_step(
        self,
        x_pos: np.ndarray,
        x_neg_list: list[np.ndarray],
    ) -> tuple[float, float, float]:
        """Training step with one positive and multiple negative inputs.

        The loss pushes positive goodness above threshold and negative
        goodness below. Using multiple negatives per positive gives a
        richer gradient signal — each negative represents a different
        "wrong hypothesis" the network must learn to reject.

        Args:
            x_pos: Positive inputs (correct labels), shape (batch, dim).
            x_neg_list: List of negative input arrays (wrong labels).

        Returns:
            Tuple of (loss, mean_pos_goodness, mean_neg_goodness).
        """
        batch_n = x_pos.shape[0]
        theta = self.threshold
        output_dim = self.w.shape[1]
        n_neg = len(x_neg_list)

        # --- Positive forward ---
        x_pos_norm = x_pos / (np.linalg.norm(x_pos, axis=1, keepdims=True) + 1e-8)
        z_pos = x_pos_norm @ self.w + self.b
        a_pos = np.maximum(0, z_pos)
        g_pos = np.mean(a_pos**2, axis=1)

        logit_pos = -g_pos + theta
        loss_pos = np.log(1 + np.exp(np.clip(logit_pos, -20, 20)))

        # Positive gradient: d(loss)/d(g_pos) = -sigmoid(-g_pos + theta)
        dsig_pos = -1.0 / (1 + np.exp(-np.clip(logit_pos, -20, 20)))
        da_pos = (dsig_pos[:, None] * 2 * a_pos / output_dim) * (z_pos > 0).astype(
            np.float64
        )
        dw = x_pos_norm.T @ da_pos / batch_n
        db = np.mean(da_pos, axis=0)

        # --- Multiple negatives: average gradients ---
        total_loss_neg = 0.0
        total_g_neg = 0.0

        for x_neg in x_neg_list:
            x_neg_norm = x_neg / (
                np.linalg.norm(x_neg, axis=1, keepdims=True) + 1e-8
            )
            z_neg = x_neg_norm @ self.w + self.b
            a_neg = np.maximum(0, z_neg)
            g_neg = np.mean(a_neg**2, axis=1)

            logit_neg = g_neg - theta
            loss_neg = np.log(1 + np.exp(np.clip(logit_neg, -20, 20)))
            total_loss_neg += float(np.mean(loss_neg))
            total_g_neg += float(np.mean(g_neg))

            # Negative gradient: d(loss)/d(g_neg) = sigmoid(g_neg - theta)
            dsig_neg = 1.0 / (1 + np.exp(-np.clip(logit_neg, -20, 20)))
            da_neg = (dsig_neg[:, None] * 2 * a_neg / output_dim) * (
                z_neg > 0
            ).astype(np.float64)
            dw += x_neg_norm.T @ da_neg / batch_n / n_neg
            db += np.mean(da_neg, axis=0) / n_neg

        loss = float(np.mean(loss_pos)) + total_loss_neg / n_neg

        # Adam update
        self.w = self._opt_w.step(self.w, dw)
        self.b = self._opt_b.step(self.b, db)

        return loss, float(np.mean(g_pos)), total_g_neg / n_neg


class FFEnhanced(MNISTApproach):
    """Enhanced Forward-Forward with wider layers and learning rate scheduling.

    Key differences from base ForwardForward:
    - 2×1000 hidden layers (2× wider — more goodness separation capacity)
    - Cosine LR schedule with warmup (converges faster in limited epochs)
    - AdamW weight decay (prevents overfitting to specific negatives)
    - 3 negatives per positive (richer per-step gradient signal)
    - He initialization (better variance preservation for ReLU)
    - Larger mini-batches (2048 — smoother gradients for higher base LR)

    Biological analogues:
    - Wider layers: V1 cortex has ~200M neurons, not 500
    - LR schedule: metaplasticity — learning rate decreases as synapses
      consolidate (BCM sliding threshold)
    - Multi-negative: brain compares multiple hypotheses simultaneously
      (predictive coding framework — Rao & Ballard 1999)
    - Weight decay: synaptic protein degradation / turnover
    """

    name = "ff_enhanced"
    uses_backprop = False

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        threshold: float = 2.0,
        lr_max: float = 0.06,
        lr_min: float = 0.001,
        epochs: int = 25,
        batch_size: int = 2048,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        n_negatives: int = 3,
        seed: int = 42,
        log_every: int = 5,
    ) -> None:
        self.hidden_sizes = hidden_sizes or [1000, 1000]
        self.threshold = threshold
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.n_negatives = n_negatives
        self.rng = np.random.default_rng(seed)
        self.log_every = log_every

        self._layers: list[_FFLayerWide] = []

    def _quick_accuracy(
        self, images: np.ndarray, labels: np.ndarray, n_eval: int = 2000
    ) -> float:
        """Evaluate accuracy on a random subset for progress tracking.

        Args:
            images: Full image set.
            labels: Full label set.
            n_eval: Number of samples to evaluate.

        Returns:
            Accuracy on the subset.
        """
        idx = self.rng.choice(
            len(images), size=min(n_eval, len(images)), replace=False
        )
        preds = self.predict(images[idx])
        return float(np.mean(preds == labels[idx]))

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train each layer greedily with enhanced Forward-Forward.

        For each layer:
        1. Create positive samples (image + correct label overlay)
        2. Create multiple negative samples (image + different wrong labels)
        3. Train the layer to push positive goodness up, negative goodness down
        4. Freeze the layer and pass data through for the next layer

        Args:
            images: Training images, shape (N, 784), float64 in [0, 1].
            labels: Training labels, shape (N,), int in [0, 9].
        """
        n_samples = images.shape[0]
        n_batches_per_epoch = max(1, n_samples // self.batch_size)
        total_steps = n_batches_per_epoch * self.epochs
        warmup_steps = n_batches_per_epoch * self.warmup_epochs

        # Build layers
        dims = [images.shape[1]] + self.hidden_sizes
        self._layers = []
        for i in range(len(self.hidden_sizes)):
            layer = _FFLayerWide(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                threshold=self.threshold,
                lr_max=self.lr_max,
                lr_min=self.lr_min,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                weight_decay=self.weight_decay,
                rng=self.rng,
            )
            self._layers.append(layer)

        global_epoch = 0

        for layer_idx, layer in enumerate(self._layers):
            print(
                f"    Layer {layer_idx + 1}/{len(self._layers)} "
                f"({dims[layer_idx]}→{dims[layer_idx + 1]})"
            )

            for epoch in range(self.epochs):
                perm = self.rng.permutation(n_samples)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n_samples, self.batch_size):
                    idx = perm[start : start + self.batch_size]

                    # Positive: correct labels
                    x_pos = _overlay_labels(images[idx], labels[idx])

                    # Multiple negatives: different wrong labels each
                    x_neg_list = []
                    for _ in range(self.n_negatives):
                        wrong = _random_wrong_labels(labels[idx], 10, self.rng)
                        x_neg_list.append(_overlay_labels(images[idx], wrong))

                    # Propagate through frozen preceding layers
                    for prev_layer in self._layers[:layer_idx]:
                        x_pos = prev_layer.forward(x_pos)
                        x_neg_list = [prev_layer.forward(xn) for xn in x_neg_list]

                    loss, _, _ = layer.train_step(x_pos, x_neg_list)
                    epoch_loss += loss
                    n_batches += 1

                global_epoch += 1
                avg_loss = epoch_loss / max(n_batches, 1)

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
                        f"      Epoch {epoch + 1}/{self.epochs} — "
                        f"loss: {avg_loss:.4f}, acc: {train_acc:.4f}"
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
        """Expose layer weights for analysis.

        Returns:
            Dict with weight matrices and biases per layer.
        """
        internals: dict[str, object] = {}
        for i, layer in enumerate(self._layers):
            internals[f"w{i + 1}"] = layer.w
            internals[f"b{i + 1}"] = layer.b
        return internals
