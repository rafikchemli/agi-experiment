"""FISTA-accelerated energy-based sparse coding with increased capacity.

Builds on v9 (augmentation + incoherence) with two key changes:

1. **FISTA inference** — Fast ISTA with Nesterov momentum for sparse code
   inference. Standard ISTA converges at O(1/t), FISTA at O(1/t²). Better
   codes mean more accurate dictionary updates and richer learned features.

2. **Increased capacity** — 300 features per class (up from 200) and 60
   settling iterations (up from 40). Previously this caused overfitting
   (v6 with 300 features got 96.0% vs 96.4% baseline), but microsaccade
   augmentation now prevents overfitting to exact pixel positions.

Architecture: Same as v9 — 10 class-specific dictionaries with incoherence
    regularization and microsaccade augmentation during training. But with
    FISTA inference instead of ISTA.

FISTA inference (per dictionary):
    z_0 = 0, y_0 = 0, t_0 = 1
    for i in 1..T:
        residual = x - y_{i-1} @ D^T
        drive = residual @ D
        z_i = soft_threshold(y_{i-1} + step * drive, lambda * step)
        t_i = (1 + sqrt(1 + 4*t_{i-1}^2)) / 2
        y_i = z_i + ((t_{i-1} - 1) / t_i) * (z_i - z_{i-1})
    return z_T

Biological analogues:
    - FISTA momentum ≈ neural adaptation / momentum in cortical dynamics.
      When a neuron's activity changes, it doesn't instantly settle — it
      overshoots slightly (momentum) then corrects. This accelerates
      convergence to stable patterns (Shriki et al. 2003).
    - The momentum coefficient (t_{i-1} - 1) / t_i grows from 0 to ~1,
      matching how neural adaptation builds up over the first ~50ms of
      stimulus presentation (Carandini & Ferster 1997).
    - 300 features per class ≈ richer cortical column representations in IT
    - 60 settling iterations ≈ ~150ms cortical recurrence (at ~2.5ms/cycle)

Based on:
    - Beck & Teboulle (2009): Fast Iterative Shrinkage-Thresholding Algorithm
    - v9 architecture (v6 + microsaccade augmentation)
    - Shriki et al. (2003): Rate models for recurrent cortical networks
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


def _random_shift(
    images: np.ndarray, max_shift: int, rng: np.random.Generator
) -> np.ndarray:
    """Apply random pixel shifts to a batch of 28x28 images.

    Args:
        images: Flattened images, shape (B, 784).
        max_shift: Maximum shift in pixels (each direction).
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


class SparseCodingV11FISTA(MNISTApproach):
    """FISTA-accelerated energy-based sparse coding with increased capacity.

    Same training pipeline as v9 but with FISTA inference for faster
    convergence to better sparse codes, and increased capacity (more
    features, more settling iterations).

    Args:
        n_features_per_class: Dictionary atoms per class.
        n_classes: Number of digit classes.
        n_settle: FISTA iterations per settling run.
        sparsity: Soft-threshold lambda.
        infer_rate: FISTA step size.
        learn_rate: Dictionary learning rate.
        incoherence_rate: Inter-dictionary repulsion strength.
        max_shift: Maximum augmentation shift in pixels.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        seed: Random seed.
    """

    name = "sparse_coding_v11"
    uses_backprop = False

    def __init__(
        self,
        n_features_per_class: int = 300,
        n_classes: int = 10,
        n_settle: int = 60,
        sparsity: float = 0.01,
        infer_rate: float = 0.1,
        learn_rate: float = 0.01,
        incoherence_rate: float = 0.001,
        max_shift: int = 1,
        epochs: int = 35,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.n_features_per_class = n_features_per_class
        self.n_classes = n_classes
        self.n_settle = n_settle
        self.sparsity = sparsity
        self.infer_rate = infer_rate
        self.learn_rate = learn_rate
        self.incoherence_rate = incoherence_rate
        self.max_shift = max_shift
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.dictionaries: list[np.ndarray] = []

    def _settle_fista(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """FISTA settling — ISTA with Nesterov momentum.

        Uses the standard FISTA algorithm (Beck & Teboulle 2009) which
        converges at O(1/t²) vs ISTA's O(1/t). The momentum coefficient
        grows from 0 to ~1 over iterations, matching neural adaptation
        dynamics.

        Args:
            x: Input, shape (B, 784).
            d: Dictionary, shape (784, n_features_per_class).

        Returns:
            Sparse codes z, shape (B, n_features_per_class).
        """
        b = x.shape[0]
        n_feat = d.shape[1]
        step = self.infer_rate
        threshold = self.sparsity * step

        z = np.zeros((b, n_feat), dtype=np.float64)
        z_prev = np.zeros_like(z)
        y = np.zeros_like(z)
        t = 1.0

        for _ in range(self.n_settle):
            # FISTA step: use momentum point y for gradient
            residual = x - y @ d.T
            drive = residual @ d
            z_new = y + step * drive
            z_new = np.maximum(0.0, z_new - threshold)
            np.minimum(z_new, 5.0, out=z_new)

            # Update momentum
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_new
            y = z_new + momentum * (z_new - z)

            # Non-negativity on momentum point too
            np.maximum(y, 0.0, out=y)

            z = z_new
            t = t_new

        return z

    def _recon_error(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error.

        Args:
            x: Input, shape (B, 784).
            d: Dictionary, shape (784, n_features_per_class).

        Returns:
            Error per sample, shape (B,).
        """
        z = self._settle_fista(x, d)
        recon = z @ d.T
        return np.mean((x - recon) ** 2, axis=1)

    def _apply_incoherence(self, k: int) -> None:
        """Push dictionary k away from other dictionaries' subspaces.

        Args:
            k: Index of the dictionary to regularize.
        """
        d_k = self.dictionaries[k]
        penalty = np.zeros_like(d_k)

        for j in range(self.n_classes):
            if j == k:
                continue
            d_j = self.dictionaries[j]
            overlap = d_j @ (d_j.T @ d_k)
            penalty += overlap

        self.dictionaries[k] -= self.incoherence_rate * penalty

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train with FISTA inference, augmentation, and incoherence.

        Args:
            images: Training images, shape (N, 784), float64 in [0, 1].
            labels: Training labels, shape (N,), int in [0, 9].
        """
        n_samples, n_px = images.shape

        # Initialize dictionaries
        self.dictionaries = []
        for _ in range(self.n_classes):
            d = self.rng.normal(0, 1.0, (n_px, self.n_features_per_class))
            norms = np.linalg.norm(d, axis=0, keepdims=True) + 1e-8
            d /= norms
            self.dictionaries.append(d)

        for epoch in range(self.epochs):
            perm = self.rng.permutation(n_samples)
            total_recon = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_batch = images[idx]
                y_batch = labels[idx]

                # Microsaccade augmentation
                x_aug = _random_shift(x_batch, self.max_shift, self.rng)

                for k in range(self.n_classes):
                    mask = y_batch == k
                    if mask.sum() < 2:  # noqa: PLR2004
                        continue

                    x_k = x_aug[mask]
                    bs = x_k.shape[0]
                    d = self.dictionaries[k]

                    z = self._settle_fista(x_k, d)

                    residual = x_k - z @ d.T
                    total_recon += float(np.sum(residual**2)) / n_px

                    self.dictionaries[k] += self.learn_rate * (residual.T @ z) / bs

                    self._apply_incoherence(k)

                    norms = np.linalg.norm(
                        self.dictionaries[k], axis=0, keepdims=True
                    )
                    self.dictionaries[k] /= norms + 1e-8

                n_batches += 1

            # Evaluate on original images
            eval_idx = self.rng.choice(
                n_samples, size=min(2000, n_samples), replace=False
            )
            preds = self.predict(images[eval_idx])
            acc = float(np.mean(preds == labels[eval_idx]))
            avg_recon = total_recon / n_samples

            self.history.append(
                EpochMetrics(epoch=epoch + 1, train_acc=acc, loss=avg_recon)
            )
            print(
                f"    Epoch {epoch + 1}/{self.epochs} — "
                f"recon: {avg_recon:.4f}, train acc: {acc:.4f}"
            )

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Classify by competitive reconstruction.

        Args:
            images: Test images, shape (N, 784).

        Returns:
            Predicted labels, shape (N,), int in [0, 9].
        """
        if not self.dictionaries:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        n = images.shape[0]
        errors = np.zeros((n, self.n_classes), dtype=np.float64)

        batch_sz = 1000
        for start in range(0, n, batch_sz):
            x_batch = images[start : start + batch_sz]
            bs = x_batch.shape[0]

            for k in range(self.n_classes):
                err = self._recon_error(x_batch, self.dictionaries[k])
                errors[start : start + bs, k] = err

        return np.argmin(errors, axis=1).astype(np.uint8)

    def get_internals(self) -> dict[str, object]:
        """Expose class-specific dictionaries for analysis.

        Returns:
            Dict with per-class dictionaries and coherence.
        """
        internals: dict[str, object] = {}
        for k, d in enumerate(self.dictionaries):
            internals[f"dictionary_{k}"] = d

        if len(self.dictionaries) >= 2:  # noqa: PLR2004
            coherences = []
            for k in range(self.n_classes):
                for j in range(k + 1, self.n_classes):
                    coh = np.linalg.norm(
                        self.dictionaries[k].T @ self.dictionaries[j]
                    )
                    coherences.append(coh)
            internals["mean_coherence"] = float(np.mean(coherences))

        return internals
