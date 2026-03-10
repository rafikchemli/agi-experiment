"""Hybrid FF + sparse coding with confidence-based arbitration.

Combines two complementary non-backprop approaches:
1. Forward-Forward (nonlinear contrastive learning) — goodness-based
2. Sparse coding v9 (linear reconstruction + incoherence) — error-based

Each system makes independent predictions. When they agree (94% of the
time), use the shared prediction. When they disagree, use the system
with higher confidence (larger decision margin).

Error analysis on MNIST test set:
    FF errors:  594 (5.9%)
    SC errors:  329 (3.3%)
    Both wrong: 195 (1.9%) — only 33% overlap
    Perfect combination ceiling: 98.0%

When they disagree (601 cases):
    SC right: 399 (66.4%)
    FF right: 134 (22.3%)
    Neither:   68 (11.3%)

A confidence-based arbiter can capture most of the 1.3% potential
improvement over SC alone (96.7% → up to 98.0%).

Confidence metrics:
    FF confidence:  goodness_gap = best_goodness - 2nd_best_goodness
    SC confidence:  error_gap = 2nd_lowest_error - lowest_error

Higher confidence = larger gap between best and second-best option.

Biological analogue:
    - Multi-stream integration: the brain processes visual information
      through multiple parallel streams (ventral for identity, dorsal
      for spatial) and combines them based on reliability (Ernst & Banks
      2002, Knill & Pouget 2004).
    - Confidence = signal-to-noise ratio of each processing stream.
      When the ventral stream is noisy (ambiguous object identity), the
      brain relies more on dorsal stream cues, and vice versa.
    - This combination is implemented via Bayesian cue combination in
      parietal cortex (Fetsch et al. 2013).
    - No backprop: each component trains independently with local rules.
      The combination is purely at the decision level (no weight sharing).

Based on:
    - Forward-Forward: Hinton (2022)
    - Sparse coding v9: Olshausen & Field (1996) + incoherence + augmentation
    - Ernst & Banks (2002): Humans integrate visual and haptic information
      in a statistically optimal fashion
    - Knill & Pouget (2004): The Bayesian brain
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


# --- Forward-Forward components ---

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


class _AdamState:
    """Per-parameter Adam optimizer state."""

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
        """Apply one Adam update.

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
    """A single Forward-Forward layer."""

    def __init__(
        self, input_dim: int, output_dim: int, threshold: float,
        lr: float, rng: np.random.Generator,
    ) -> None:
        self.threshold = threshold
        scale = np.sqrt(2.0 / input_dim)
        self.w = rng.normal(0, scale, (input_dim, output_dim))
        self.b = np.zeros(output_dim, dtype=np.float64)
        self._opt_w = _AdamState(self.w.shape, lr=lr)
        self._opt_b = _AdamState(self.b.shape, lr=lr)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: L2-normalize, linear, ReLU.

        Args:
            x: Input, shape (batch, input_dim).

        Returns:
            Activations, shape (batch, output_dim).
        """
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return np.maximum(0, x_norm @ self.w + self.b)

    def goodness(self, activations: np.ndarray) -> np.ndarray:
        """Goodness = mean of squared activations.

        Args:
            activations: Shape (batch, output_dim).

        Returns:
            Shape (batch,).
        """
        return np.mean(activations**2, axis=1)

    def train_step(
        self, x_pos: np.ndarray, x_neg: np.ndarray
    ) -> tuple[float, float, float]:
        """One training step.

        Args:
            x_pos: Positive inputs, shape (batch, input_dim).
            x_neg: Negative inputs, shape (batch, input_dim).

        Returns:
            Tuple of (loss, mean_pos_goodness, mean_neg_goodness).
        """
        batch_n = x_pos.shape[0]
        theta = self.threshold
        output_dim = self.w.shape[1]

        x_pos_norm = x_pos / (np.linalg.norm(x_pos, axis=1, keepdims=True) + 1e-8)
        z_pos = x_pos_norm @ self.w + self.b
        a_pos = np.maximum(0, z_pos)
        g_pos = np.mean(a_pos**2, axis=1)

        x_neg_norm = x_neg / (np.linalg.norm(x_neg, axis=1, keepdims=True) + 1e-8)
        z_neg = x_neg_norm @ self.w + self.b
        a_neg = np.maximum(0, z_neg)
        g_neg = np.mean(a_neg**2, axis=1)

        logit_pos = -g_pos + theta
        logit_neg = g_neg - theta
        loss_pos = np.log(1 + np.exp(np.clip(logit_pos, -20, 20)))
        loss_neg = np.log(1 + np.exp(np.clip(logit_neg, -20, 20)))
        loss = float(np.mean(loss_pos + loss_neg))

        dsig_pos = -1.0 / (1 + np.exp(-np.clip(logit_pos, -20, 20)))
        da_pos = (dsig_pos[:, None] * 2 * a_pos / output_dim) * (z_pos > 0).astype(np.float64)
        dw_pos = x_pos_norm.T @ da_pos / batch_n
        db_pos = np.mean(da_pos, axis=0)

        dsig_neg = 1.0 / (1 + np.exp(-np.clip(logit_neg, -20, 20)))
        da_neg = (dsig_neg[:, None] * 2 * a_neg / output_dim) * (z_neg > 0).astype(np.float64)
        dw_neg = x_neg_norm.T @ da_neg / batch_n
        db_neg = np.mean(da_neg, axis=0)

        self.w = self._opt_w.step(self.w, dw_pos + dw_neg)
        self.b = self._opt_b.step(self.b, db_pos + db_neg)

        return loss, float(np.mean(g_pos)), float(np.mean(g_neg))


# --- Sparse Coding components ---

def _random_shift(
    images: np.ndarray, max_shift: int, rng: np.random.Generator
) -> np.ndarray:
    """Apply random pixel shifts for microsaccade augmentation.

    Args:
        images: Flattened images, shape (B, 784).
        max_shift: Maximum shift in pixels.
        rng: Random generator.

    Returns:
        Shifted images, shape (B, 784).
    """
    b = images.shape[0]
    imgs_2d = images.reshape(b, 28, 28)
    shifted = np.zeros_like(imgs_2d)
    dx = rng.integers(-max_shift, max_shift + 1, size=b)
    dy = rng.integers(-max_shift, max_shift + 1, size=b)
    for i in range(b):
        sx, sy = int(dx[i]), int(dy[i])
        src_y = slice(max(0, -sy), min(28, 28 - sy))
        src_x = slice(max(0, -sx), min(28, 28 - sx))
        dst_y = slice(max(0, sy), min(28, 28 + sy))
        dst_x = slice(max(0, sx), min(28, 28 + sx))
        shifted[i, dst_y, dst_x] = imgs_2d[i, src_y, src_x]
    return shifted.reshape(b, 784)


class HybridV17FFSC(MNISTApproach):
    """Hybrid FF + Sparse Coding with confidence-based arbitration.

    Trains both systems independently (both non-backprop), then combines
    their predictions at test time using confidence-weighted voting.

    Args:
        ff_epochs: FF training epochs per layer.
        ff_lr: FF Adam learning rate.
        ff_hidden: FF hidden layer sizes.
        ff_threshold: FF goodness threshold.
        sc_features: SC dictionary atoms per class.
        sc_settle: SC ISTA iterations.
        sc_sparsity: SC soft-threshold lambda.
        sc_lr: SC dictionary learning rate.
        sc_incoherence: SC incoherence rate.
        sc_epochs: SC training epochs.
        sc_shift: SC augmentation shift.
        batch_size: Mini-batch size for both.
        seed: Random seed.
    """

    name = "hybrid_v17"
    uses_backprop = False

    def __init__(
        self,
        ff_epochs: int = 25,
        ff_lr: float = 0.03,
        ff_hidden: list[int] | None = None,
        ff_threshold: float = 2.0,
        sc_features: int = 200,
        sc_settle: int = 40,
        sc_sparsity: float = 0.01,
        sc_lr: float = 0.01,
        sc_incoherence: float = 0.001,
        sc_epochs: int = 35,
        sc_shift: int = 1,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.ff_epochs = ff_epochs
        self.ff_lr = ff_lr
        self.ff_hidden = ff_hidden or [500, 500]
        self.ff_threshold = ff_threshold
        self.sc_features = sc_features
        self.sc_settle = sc_settle
        self.sc_sparsity = sc_sparsity
        self.sc_lr = sc_lr
        self.sc_incoherence = sc_incoherence
        self.sc_epochs = sc_epochs
        self.sc_shift = sc_shift
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self._ff_layers: list[_FFLayer] = []
        self._sc_dicts: list[np.ndarray] = []
        self.n_classes = 10

    # --- SC helper methods ---

    def _settle(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """ISTA settling.

        Args:
            x: Input, shape (B, 784).
            d: Dictionary, shape (784, features).

        Returns:
            Sparse codes, shape (B, features).
        """
        b = x.shape[0]
        n_feat = d.shape[1]
        z = np.zeros((b, n_feat), dtype=np.float64)
        step = 0.1
        threshold = self.sc_sparsity * step
        for _ in range(self.sc_settle):
            residual = x - z @ d.T
            drive = residual @ d
            z = z + step * drive
            z = np.maximum(0.0, z - threshold)
            np.minimum(z, 5.0, out=z)
        return z

    def _sc_errors(self, images: np.ndarray) -> np.ndarray:
        """Compute SC reconstruction errors against all dictionaries.

        Args:
            images: Input, shape (N, 784).

        Returns:
            Errors, shape (N, 10).
        """
        n = images.shape[0]
        errors = np.zeros((n, self.n_classes), dtype=np.float64)
        batch_sz = 1000
        for start in range(0, n, batch_sz):
            x_batch = images[start : start + batch_sz]
            bs = x_batch.shape[0]
            for k in range(self.n_classes):
                z = self._settle(x_batch, self._sc_dicts[k])
                recon = z @ self._sc_dicts[k].T
                errors[start : start + bs, k] = np.mean(
                    (x_batch - recon) ** 2, axis=1
                )
        return errors

    def _ff_goodness_all(self, images: np.ndarray) -> np.ndarray:
        """Compute FF goodness for all 10 label hypotheses.

        Args:
            images: Input, shape (N, 784).

        Returns:
            Goodness, shape (N, 10).
        """
        n = images.shape[0]
        goodness = np.zeros((n, 10), dtype=np.float64)
        for label in range(10):
            label_vec = np.full(n, label, dtype=np.intp)
            x = _overlay_labels(images, label_vec)
            total_g = np.zeros(n, dtype=np.float64)
            for layer in self._ff_layers:
                x = layer.forward(x)
                total_g += layer.goodness(x)
            goodness[:, label] = total_g
        return goodness

    def _apply_incoherence(self, k: int) -> None:
        """Push SC dictionary k away from others.

        Args:
            k: Dictionary index.
        """
        d_k = self._sc_dicts[k]
        penalty = np.zeros_like(d_k)
        for j in range(self.n_classes):
            if j == k:
                continue
            d_j = self._sc_dicts[j]
            overlap = d_j @ (d_j.T @ d_k)
            penalty += overlap
        self._sc_dicts[k] -= self.sc_incoherence * penalty

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train both FF and SC independently.

        Args:
            images: Training images, shape (N, 784).
            labels: Training labels, shape (N,).
        """
        n_samples, n_px = images.shape

        # --- Train FF ---
        print("    === Training Forward-Forward ===")
        dims = [n_px] + self.ff_hidden
        self._ff_layers = []
        for i in range(len(self.ff_hidden)):
            layer = _FFLayer(dims[i], dims[i + 1], self.ff_threshold,
                             self.ff_lr, self.rng)
            self._ff_layers.append(layer)

        for layer_idx, layer in enumerate(self._ff_layers):
            print(f"    FF Layer {layer_idx+1}/{len(self._ff_layers)}")
            for epoch in range(self.ff_epochs):
                perm = self.rng.permutation(n_samples)
                for start in range(0, n_samples, self.batch_size):
                    idx = perm[start : start + self.batch_size]
                    x_pos = _overlay_labels(images[idx], labels[idx])
                    wrong = _random_wrong_labels(labels[idx], 10, self.rng)
                    x_neg = _overlay_labels(images[idx], wrong)
                    for prev_layer in self._ff_layers[:layer_idx]:
                        x_pos = prev_layer.forward(x_pos)
                        x_neg = prev_layer.forward(x_neg)
                    layer.train_step(x_pos, x_neg)
                if (epoch + 1) % 10 == 0:
                    print(f"      Epoch {epoch+1}/{self.ff_epochs}")

        # --- Train SC ---
        print("    === Training Sparse Coding ===")
        self._sc_dicts = []
        for _ in range(self.n_classes):
            d = self.rng.normal(0, 1.0, (n_px, self.sc_features))
            d /= np.linalg.norm(d, axis=0, keepdims=True) + 1e-8
            self._sc_dicts.append(d)

        for epoch in range(self.sc_epochs):
            perm = self.rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_batch = images[idx]
                y_batch = labels[idx]
                x_aug = _random_shift(x_batch, self.sc_shift, self.rng)
                for k in range(self.n_classes):
                    mask = y_batch == k
                    if mask.sum() < 2:  # noqa: PLR2004
                        continue
                    x_k = x_aug[mask]
                    bs = x_k.shape[0]
                    z = self._settle(x_k, self._sc_dicts[k])
                    residual = x_k - z @ self._sc_dicts[k].T
                    self._sc_dicts[k] += self.sc_lr * (residual.T @ z) / bs
                    self._apply_incoherence(k)
                    norms = np.linalg.norm(self._sc_dicts[k], axis=0, keepdims=True)
                    self._sc_dicts[k] /= norms + 1e-8

            if (epoch + 1) % 10 == 0:
                eval_idx = self.rng.choice(n_samples, size=min(2000, n_samples), replace=False)
                preds = self.predict(images[eval_idx])
                acc = float(np.mean(preds == labels[eval_idx]))
                self.history.append(
                    EpochMetrics(epoch=epoch + 1, train_acc=acc, loss=0.0)
                )
                print(f"    SC Epoch {epoch+1}/{self.sc_epochs} — acc: {acc:.4f}")

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict using confidence-weighted combination of FF and SC.

        For each image:
        1. Get FF prediction (argmax goodness) and confidence (goodness gap)
        2. Get SC prediction (argmin error) and confidence (error gap)
        3. If they agree: use the shared prediction
        4. If they disagree: use the more confident system

        Confidence is normalized by dividing by the mean of the best score
        to make FF goodness and SC error comparable.

        Args:
            images: Test images, shape (N, 784).

        Returns:
            Predicted labels, shape (N,).
        """
        if not self._ff_layers or not self._sc_dicts:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        n = images.shape[0]

        # FF predictions and confidence
        goodness = self._ff_goodness_all(images)  # (N, 10)
        ff_preds = np.argmax(goodness, axis=1)
        # Confidence: gap between best and second-best goodness
        sorted_g = np.sort(goodness, axis=1)
        ff_confidence = sorted_g[:, -1] - sorted_g[:, -2]
        # Normalize: divide by mean of best goodness
        ff_conf_norm = ff_confidence / (np.mean(sorted_g[:, -1]) + 1e-8)

        # SC predictions and confidence
        errors = self._sc_errors(images)  # (N, 10)
        sc_preds = np.argmin(errors, axis=1)
        # Confidence: gap between 2nd-lowest and lowest error
        sorted_e = np.sort(errors, axis=1)
        sc_confidence = sorted_e[:, 1] - sorted_e[:, 0]
        # Normalize: divide by mean of lowest error
        sc_conf_norm = sc_confidence / (np.mean(sorted_e[:, 0]) + 1e-8)

        # Combine: agree → shared, disagree → higher confidence
        agree = (ff_preds == sc_preds)
        result = np.where(agree, sc_preds, np.uint8(0))

        # For disagreements, trust the more confident system
        disagree = ~agree
        trust_ff = disagree & (ff_conf_norm > sc_conf_norm)
        trust_sc = disagree & ~trust_ff
        result[trust_ff] = ff_preds[trust_ff]
        result[trust_sc] = sc_preds[trust_sc]

        return result.astype(np.uint8)

    def get_internals(self) -> dict[str, object]:
        """Expose internals from both systems.

        Returns:
            Dict with FF weights and SC dictionaries.
        """
        internals: dict[str, object] = {}
        for i, layer in enumerate(self._ff_layers):
            internals[f"ff_w{i+1}"] = layer.w
        for k, d in enumerate(self._sc_dicts):
            internals[f"sc_dict_{k}"] = d
        return internals
