"""Incoherent energy-based sparse coding with data augmentation.

Builds on v6 (class-specific dictionaries + incoherence regularization)
and adds data augmentation to close the generalization gap. v6 achieves
98.4% train accuracy but only 96.4% test accuracy — a 2% gap caused by
overfitting to exact pixel positions.

Data augmentation provides slightly different views of each digit per
epoch, forcing dictionaries to learn position-robust features.

Architecture: Same as v6 — 10 class-specific dictionaries with
    incoherence regularization between them.

Data augmentation:
    - Random pixel shifts ±2 in x and y (4 directions)
    - Applied each epoch, so each image appears slightly different

Biological analogue:
    - Microsaccades: the eye constantly makes small involuntary movements
      (1-2 pixels at retinal resolution), so the retinal image is never
      exactly the same twice. This is essential for preventing adaptation
      and learning position-invariant features (Martinez-Conde et al. 2004)
    - The brain never sees the exact same retinal image twice — slight
      jitter is the natural state of visual input

Based on:
    - v6 architecture + Ramirez et al. (2010) incoherence
    - Martinez-Conde et al. (2004): Microsaccades and visual perception
    - Wallis & Rolls (1997): Invariant recognition via transformed views
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


def _random_shift(
    images: np.ndarray, max_shift: int, rng: np.random.Generator
) -> np.ndarray:
    """Apply random pixel shifts to a batch of 28x28 images.

    Simulates microsaccades — small involuntary eye movements that shift
    the retinal image by 1-2 pixels.

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
        # Source and destination slices
        src_y = slice(max(0, -sy), min(28, 28 - sy))
        src_x = slice(max(0, -sx), min(28, 28 - sx))
        dst_y = slice(max(0, sy), min(28, 28 + sy))
        dst_x = slice(max(0, sx), min(28, 28 + sx))
        shifted[i, dst_y, dst_x] = imgs_2d[i, src_y, src_x]

    return shifted.reshape(b, 784)


class SparseCodingV9Augmented(MNISTApproach):
    """Incoherent energy-based sparse coding with microsaccade augmentation.

    Same architecture as v6 but with random pixel shifts during training
    to close the generalization gap.

    Args:
        n_features_per_class: Dictionary atoms per class.
        n_classes: Number of digit classes.
        n_settle: ISTA iterations per settling run.
        sparsity: Soft-threshold λ.
        infer_rate: ISTA step size.
        learn_rate: Dictionary learning rate.
        incoherence_rate: Inter-dictionary repulsion strength.
        max_shift: Maximum augmentation shift in pixels.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        seed: Random seed.
    """

    name = "sparse_coding_v9"
    uses_backprop = False

    def __init__(
        self,
        n_features_per_class: int = 200,
        n_classes: int = 10,
        n_settle: int = 40,
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

    def _settle(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """ISTA settling for a specific dictionary.

        Args:
            x: Input, shape (B, 784).
            d: Dictionary, shape (784, n_features_per_class).

        Returns:
            Sparse codes z, shape (B, n_features_per_class).
        """
        b = x.shape[0]
        n_feat = d.shape[1]
        z = np.zeros((b, n_feat), dtype=np.float64)
        step = self.infer_rate
        threshold = self.sparsity * step

        for _ in range(self.n_settle):
            residual = x - z @ d.T
            drive = residual @ d
            z = z + step * drive
            z = np.maximum(0.0, z - threshold)
            np.minimum(z, 5.0, out=z)

        return z

    def _recon_error(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error.

        Args:
            x: Input, shape (B, 784).
            d: Dictionary, shape (784, n_features_per_class).

        Returns:
            Error per sample, shape (B,).
        """
        z = self._settle(x, d)
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
        """Train with data augmentation + incoherence regularization.

        Each epoch applies random pixel shifts (microsaccades) to training
        images before dictionary learning. This prevents overfitting to
        exact pixel positions and forces robust feature learning.

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

                # Microsaccade augmentation: random shift each image
                x_aug = _random_shift(x_batch, self.max_shift, self.rng)

                for k in range(self.n_classes):
                    mask = y_batch == k
                    if mask.sum() < 2:  # noqa: PLR2004
                        continue

                    x_k = x_aug[mask]
                    bs = x_k.shape[0]
                    d = self.dictionaries[k]

                    z = self._settle(x_k, d)

                    residual = x_k - z @ d.T
                    total_recon += float(np.sum(residual**2)) / n_px

                    self.dictionaries[k] += self.learn_rate * (residual.T @ z) / bs

                    self._apply_incoherence(k)

                    norms = np.linalg.norm(
                        self.dictionaries[k], axis=0, keepdims=True
                    )
                    self.dictionaries[k] /= norms + 1e-8

                n_batches += 1

            # Evaluate on ORIGINAL images (not augmented)
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
        """Classify by competitive reconstruction (no augmentation at test).

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
