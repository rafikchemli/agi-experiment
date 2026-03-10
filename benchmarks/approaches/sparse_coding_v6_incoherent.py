"""Energy-based sparse coding with dictionary incoherence regularization.

Builds on v5 (class-specific dictionaries, competitive reconstruction) but
adds a key missing ingredient: **incoherence regularization** that forces
dictionaries to span different subspaces. This prevents dictionary_3 from
accidentally explaining digit 5, without needing any contrastive/negative
examples.

Architecture:
    D_k ∈ R^(784 × n_features_per_class), k = 0..9
    For class k: x ≈ D_k @ z_k, where z_k is sparse
    Prediction: argmin_k ||x - D_k @ z_k||²

Training (per batch, per class k):
    1. Select images of class k
    2. ISTA settling: find sparse codes z_k
    3. Hebbian update: ΔD_k ∝ residual × z_kᵀ
    4. Incoherence penalty: ΔD_k -= η_inc * Σ_{j≠k} D_j @ (D_jᵀ @ D_k)
       Push D_k's atoms away from other dictionaries' subspaces
    5. Normalize columns (homeostatic scaling)

Biological analogues:
    - Class-specific dictionaries ≈ specialized cortical columns in IT cortex
    - Incoherence penalty ≈ lateral inhibition between cortical columns
      (competing columns suppress shared feature representations, forcing
      each to develop unique detectors — Kaschube et al. 2010)
    - The regularization is local: each column only needs to sense what
      neighboring columns represent, not what stimuli they receive
    - Energy-based classification ≈ analysis-by-synthesis / predictive coding
      with multiple competing hypotheses (Rao & Ballard 1999)

Based on:
    - Olshausen & Field (1996): Sparse coding
    - Wright et al. (2009): Sparse representation for classification (SRC)
    - Ramirez et al. (2010): Classification and clustering via dictionary
      learning with structured incoherence
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


class SparseCodingV6Incoherent(MNISTApproach):
    """Energy-based sparse coding with incoherence-regularized dictionaries.

    Each class gets its own dictionary. Dictionaries are regularized to
    span different subspaces via an incoherence penalty, preventing
    cross-class confusion without needing contrastive/negative examples.

    Args:
        n_features_per_class: Dictionary atoms per class.
        n_classes: Number of digit classes.
        n_settle: ISTA iterations per settling run.
        sparsity: Soft-threshold λ (lateral inhibition strength).
        infer_rate: ISTA step size.
        learn_rate: Dictionary learning rate.
        incoherence_rate: Strength of inter-dictionary repulsion.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        seed: Random seed.
    """

    name = "sparse_coding_v6"
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
        epochs: int = 30,
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.dictionaries: list[np.ndarray] = []

    def _settle(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """ISTA settling for a specific dictionary.

        Args:
            x: Input images, shape (B, 784).
            d: Dictionary for one class, shape (784, n_features_per_class).

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
        """Compute per-sample reconstruction error for a dictionary.

        Args:
            x: Input images, shape (B, 784).
            d: Dictionary, shape (784, n_features_per_class).

        Returns:
            Reconstruction error per sample, shape (B,).
        """
        z = self._settle(x, d)
        recon = z @ d.T
        return np.mean((x - recon) ** 2, axis=1)

    def _apply_incoherence(self, k: int) -> None:
        """Push dictionary k away from all other dictionaries' subspaces.

        For each other class j, compute the projection of D_k onto D_j's
        column space: D_j @ (D_jᵀ @ D_k). Subtract this from D_k, scaled
        by the incoherence rate.

        Biologically: lateral inhibition between cortical columns. Column k
        suppresses any feature representations that overlap with column j,
        forcing unique specialization.

        Args:
            k: Index of the dictionary to regularize.
        """
        d_k = self.dictionaries[k]
        penalty = np.zeros_like(d_k)

        for j in range(self.n_classes):
            if j == k:
                continue
            d_j = self.dictionaries[j]
            # Project D_k onto D_j's subspace and subtract
            # overlap = D_j @ (D_j^T @ D_k) — the component of D_k
            # that lives in D_j's column space
            overlap = d_j @ (d_j.T @ d_k)
            penalty += overlap

        self.dictionaries[k] -= self.incoherence_rate * penalty

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train incoherence-regularized class-specific dictionaries.

        For each mini-batch, for each class k:
        1. Select images belonging to class k
        2. ISTA settle to find sparse codes
        3. Hebbian dictionary update: ΔD_k ∝ residual × z_kᵀ
        4. Incoherence penalty: push D_k away from other dictionaries
        5. Normalize columns (homeostatic scaling)

        Args:
            images: Training images, shape (N, 784), float64 in [0, 1].
            labels: Training labels, shape (N,), int in [0, 9].
        """
        n_samples, n_px = images.shape

        # Initialize one dictionary per class
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

                for k in range(self.n_classes):
                    mask = y_batch == k
                    if mask.sum() < 2:  # noqa: PLR2004
                        continue

                    x_k = x_batch[mask]
                    bs = x_k.shape[0]
                    d = self.dictionaries[k]

                    # ISTA settle
                    z = self._settle(x_k, d)

                    # Reconstruction error for diagnostics
                    residual = x_k - z @ d.T
                    total_recon += float(np.sum(residual**2)) / n_px

                    # Hebbian dictionary update: ΔD ∝ residual × z
                    self.dictionaries[k] += self.learn_rate * (residual.T @ z) / bs

                    # Incoherence penalty: push away from other dictionaries
                    self._apply_incoherence(k)

                    # Homeostatic normalization
                    norms = np.linalg.norm(
                        self.dictionaries[k], axis=0, keepdims=True
                    )
                    self.dictionaries[k] /= norms + 1e-8

                n_batches += 1

            # Quick accuracy check on a subset
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
        """Classify by competitive reconstruction — lowest error wins.

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
            Dict with per-class dictionaries and inter-dictionary coherence.
        """
        internals: dict[str, object] = {}
        for k, d in enumerate(self.dictionaries):
            internals[f"dictionary_{k}"] = d

        # Measure inter-dictionary coherence (lower = more separated)
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
