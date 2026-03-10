"""Hierarchical sparse coding — V1 shared features → V2 class-specific.

The brain processes visual information hierarchically: V1 extracts universal
edge/stroke features, V2 combines them into class-specific patterns. This
architecture mirrors that hierarchy with two dictionary layers.

Architecture:
    V1 (shared): D1 ∈ R^(784 × n_v1) — universal edge detectors
        z1 = ISTA(x, D1) — sparse edge code for any image

    V2 (class-specific): D2_k ∈ R^(n_v1 × n_v2), k = 0..9
        z2_k = ISTA(z1, D2_k) — sparse combination code for class k

    Classification: argmin_k ||z1 - D2_k @ z2_k||²
        "Which class model best explains my V1 features?"

Training:
    Phase 1 (unsupervised): Learn D1 from all images via Hebbian updates
        ΔD1 ∝ residual × z1ᵀ — standard sparse dictionary learning

    Phase 2 (class-specific): Freeze D1, compute all z1, then:
        For each class k:
            1. ISTA settle z2_k = ISTA(z1_k, D2_k)
            2. Hebbian update: ΔD2_k ∝ residual × z2_kᵀ
            3. Incoherence: ΔD2_k -= η * Σ_{j≠k} D2_j @ (D2_jᵀ @ D2_k)
            4. Normalize columns

Biological analogues:
    - V1 dictionary ≈ primary visual cortex (orientation-selective cells)
    - V2 dictionary ≈ secondary visual cortex (texture/shape cells)
    - Hierarchy ≈ cortical processing stream (V1 → V2 → V4 → IT)
    - Phase 1 unsupervised ≈ early visual development (critical period)
    - Phase 2 class-specific ≈ category learning in inferotemporal cortex
    - Incoherence penalty ≈ lateral inhibition between IT columns

Based on:
    - Olshausen & Field (1996): Sparse coding in V1
    - Karklin & Lewicki (2005): Hierarchical sparse coding
    - Ramirez et al. (2010): Dictionary learning with structured incoherence
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


class SparseCodingV8Hierarchical(MNISTApproach):
    """Hierarchical sparse coding: shared V1 + class-specific V2.

    Phase 1 learns universal visual features (edges, strokes).
    Phase 2 learns class-specific combinations with incoherence.
    Classification via competitive reconstruction at the V2 level.

    Args:
        n_v1: Number of V1 dictionary atoms (shared edge detectors).
        n_v2: Number of V2 dictionary atoms per class.
        n_classes: Number of digit classes.
        n_settle_v1: ISTA iterations for V1 settling.
        n_settle_v2: ISTA iterations for V2 settling.
        sparsity_v1: V1 sparsity threshold.
        sparsity_v2: V2 sparsity threshold.
        infer_rate: ISTA step size (both layers).
        learn_rate_v1: V1 dictionary learning rate.
        learn_rate_v2: V2 dictionary learning rate.
        incoherence_rate: V2 inter-dictionary repulsion strength.
        epochs_v1: Unsupervised V1 training epochs.
        epochs_v2: Supervised V2 training epochs.
        batch_size: Mini-batch size.
        seed: Random seed.
    """

    name = "sparse_coding_v8"
    uses_backprop = False

    def __init__(
        self,
        n_v1: int = 500,
        n_v2: int = 100,
        n_classes: int = 10,
        n_settle_v1: int = 50,
        n_settle_v2: int = 30,
        sparsity_v1: float = 0.01,
        sparsity_v2: float = 0.01,
        infer_rate: float = 0.1,
        learn_rate_v1: float = 0.005,
        learn_rate_v2: float = 0.01,
        incoherence_rate: float = 0.001,
        epochs_v1: int = 15,
        epochs_v2: int = 20,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.n_v1 = n_v1
        self.n_v2 = n_v2
        self.n_classes = n_classes
        self.n_settle_v1 = n_settle_v1
        self.n_settle_v2 = n_settle_v2
        self.sparsity_v1 = sparsity_v1
        self.sparsity_v2 = sparsity_v2
        self.infer_rate = infer_rate
        self.learn_rate_v1 = learn_rate_v1
        self.learn_rate_v2 = learn_rate_v2
        self.incoherence_rate = incoherence_rate
        self.epochs_v1 = epochs_v1
        self.epochs_v2 = epochs_v2
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # Learned during train()
        self.D1: np.ndarray | None = None  # V1 dictionary (784, n_v1)
        self.D2s: list[np.ndarray] = []  # V2 dictionaries (n_v1, n_v2) × 10

    def _settle(
        self, x: np.ndarray, d: np.ndarray, n_steps: int, sparsity: float
    ) -> np.ndarray:
        """ISTA settling for any dictionary layer.

        Args:
            x: Input, shape (B, input_dim).
            d: Dictionary, shape (input_dim, n_atoms).
            n_steps: Number of ISTA iterations.
            sparsity: Soft-threshold level.

        Returns:
            Sparse codes z, shape (B, n_atoms).
        """
        b = x.shape[0]
        n_feat = d.shape[1]
        z = np.zeros((b, n_feat), dtype=np.float64)
        step = self.infer_rate
        threshold = sparsity * step

        for _ in range(n_steps):
            residual = x - z @ d.T
            drive = residual @ d
            z = z + step * drive
            z = np.maximum(0.0, z - threshold)
            np.minimum(z, 5.0, out=z)

        return z

    def _encode_v1(self, images: np.ndarray) -> np.ndarray:
        """Encode images through the frozen V1 dictionary.

        Args:
            images: Input images, shape (B, 784).

        Returns:
            V1 sparse codes, shape (B, n_v1).
        """
        assert self.D1 is not None
        return self._settle(images, self.D1, self.n_settle_v1, self.sparsity_v1)

    def _recon_error_v2(self, z1: np.ndarray, d2: np.ndarray) -> np.ndarray:
        """Compute V2 reconstruction error for a class dictionary.

        Args:
            z1: V1 codes, shape (B, n_v1).
            d2: V2 dictionary for one class, shape (n_v1, n_v2).

        Returns:
            Reconstruction error per sample, shape (B,).
        """
        z2 = self._settle(z1, d2, self.n_settle_v2, self.sparsity_v2)
        recon = z2 @ d2.T
        return np.mean((z1 - recon) ** 2, axis=1)

    def _apply_incoherence_v2(self, k: int) -> None:
        """Push V2 dictionary k away from all other V2 dictionaries.

        Args:
            k: Index of the dictionary to regularize.
        """
        d_k = self.D2s[k]
        penalty = np.zeros_like(d_k)

        for j in range(self.n_classes):
            if j == k:
                continue
            d_j = self.D2s[j]
            overlap = d_j @ (d_j.T @ d_k)
            penalty += overlap

        self.D2s[k] -= self.incoherence_rate * penalty

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train hierarchical sparse coding in two phases.

        Phase 1: Learn shared V1 dictionary (unsupervised, all images).
        Phase 2: Freeze V1, learn class-specific V2 dictionaries with
                 incoherence regularization.

        Args:
            images: Training images, shape (N, 784), float64 in [0, 1].
            labels: Training labels, shape (N,), int in [0, 9].
        """
        n_samples, n_px = images.shape

        # ── Phase 1: V1 dictionary (unsupervised) ──
        print("    Phase 1: Learning V1 shared dictionary (unsupervised)")
        self.D1 = self.rng.normal(0, 1.0, (n_px, self.n_v1))
        norms = np.linalg.norm(self.D1, axis=0, keepdims=True) + 1e-8
        self.D1 /= norms

        for epoch in range(self.epochs_v1):
            perm = self.rng.permutation(n_samples)
            total_recon = 0.0

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_batch = images[idx]
                bs = len(idx)

                z1 = self._settle(
                    x_batch, self.D1, self.n_settle_v1, self.sparsity_v1
                )
                residual = x_batch - z1 @ self.D1.T
                total_recon += float(np.sum(residual**2)) / n_px

                # Hebbian dictionary update
                self.D1 += self.learn_rate_v1 * (residual.T @ z1) / bs

                # Homeostatic normalization
                norms = np.linalg.norm(self.D1, axis=0, keepdims=True) + 1e-8
                self.D1 /= norms

            avg_recon = total_recon / n_samples
            self.history.append(
                EpochMetrics(epoch=epoch + 1, train_acc=0.0, loss=avg_recon)
            )
            print(f"      V1 Epoch {epoch + 1}/{self.epochs_v1} — recon: {avg_recon:.4f}")

        # ── Compute V1 codes for all images (frozen V1) ──
        print("    Computing V1 codes for all images...")
        z1_all = np.zeros((n_samples, self.n_v1), dtype=np.float64)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            z1_all[start:end] = self._encode_v1(images[start:end])

        # ── Phase 2: V2 class-specific dictionaries ──
        print("    Phase 2: Learning V2 class-specific dictionaries")
        self.D2s = []
        for _ in range(self.n_classes):
            d2 = self.rng.normal(0, 1.0, (self.n_v1, self.n_v2))
            norms = np.linalg.norm(d2, axis=0, keepdims=True) + 1e-8
            d2 /= norms
            self.D2s.append(d2)

        for epoch in range(self.epochs_v2):
            perm = self.rng.permutation(n_samples)
            total_recon = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                z1_batch = z1_all[idx]
                y_batch = labels[idx]

                for k in range(self.n_classes):
                    mask = y_batch == k
                    if mask.sum() < 2:  # noqa: PLR2004
                        continue

                    z1_k = z1_batch[mask]
                    bs = z1_k.shape[0]
                    d2 = self.D2s[k]

                    # ISTA settle at V2 level
                    z2 = self._settle(z1_k, d2, self.n_settle_v2, self.sparsity_v2)

                    # Reconstruction error at V2
                    residual = z1_k - z2 @ d2.T
                    total_recon += float(np.sum(residual**2)) / self.n_v1

                    # Hebbian V2 dictionary update
                    self.D2s[k] += self.learn_rate_v2 * (residual.T @ z2) / bs

                    # Incoherence penalty
                    self._apply_incoherence_v2(k)

                    # Homeostatic normalization
                    norms = np.linalg.norm(
                        self.D2s[k], axis=0, keepdims=True
                    )
                    self.D2s[k] /= norms + 1e-8

                n_batches += 1

            # Quick accuracy check
            eval_idx = self.rng.choice(
                n_samples, size=min(2000, n_samples), replace=False
            )
            preds = self._predict_from_z1(z1_all[eval_idx])
            acc = float(np.mean(preds == labels[eval_idx]))
            avg_recon = total_recon / n_samples

            global_epoch = self.epochs_v1 + epoch + 1
            self.history.append(
                EpochMetrics(epoch=global_epoch, train_acc=acc, loss=avg_recon)
            )
            print(
                f"      V2 Epoch {epoch + 1}/{self.epochs_v2} — "
                f"recon: {avg_recon:.4f}, train acc: {acc:.4f}"
            )

    def _predict_from_z1(self, z1: np.ndarray) -> np.ndarray:
        """Classify from pre-computed V1 codes.

        Args:
            z1: V1 sparse codes, shape (N, n_v1).

        Returns:
            Predicted labels, shape (N,).
        """
        n = z1.shape[0]
        errors = np.zeros((n, self.n_classes), dtype=np.float64)

        batch_sz = 1000
        for start in range(0, n, batch_sz):
            z1_batch = z1[start : start + batch_sz]
            bs = z1_batch.shape[0]

            for k in range(self.n_classes):
                err = self._recon_error_v2(z1_batch, self.D2s[k])
                errors[start : start + bs, k] = err

        return np.argmin(errors, axis=1).astype(np.uint8)

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Classify by hierarchical encoding + competitive reconstruction.

        1. Encode images through frozen V1 → sparse codes z1
        2. For each class k, settle z2_k = ISTA(z1, D2_k)
        3. Pick class with lowest V2 reconstruction error

        Args:
            images: Test images, shape (N, 784).

        Returns:
            Predicted labels, shape (N,), int in [0, 9].
        """
        if self.D1 is None or not self.D2s:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        # Encode through V1
        n = images.shape[0]
        z1 = np.zeros((n, self.n_v1), dtype=np.float64)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            z1[start:end] = self._encode_v1(images[start:end])

        return self._predict_from_z1(z1)

    def get_internals(self) -> dict[str, object]:
        """Expose V1 and V2 dictionaries for analysis.

        Returns:
            Dict with V1 dictionary, V2 class dictionaries, and coherence.
        """
        internals: dict[str, object] = {"D1": self.D1}
        for k, d2 in enumerate(self.D2s):
            internals[f"D2_{k}"] = d2

        if len(self.D2s) >= 2:  # noqa: PLR2004
            coherences = []
            for k in range(self.n_classes):
                for j in range(k + 1, self.n_classes):
                    coh = np.linalg.norm(self.D2s[k].T @ self.D2s[j])
                    coherences.append(coh)
            internals["mean_v2_coherence"] = float(np.mean(coherences))

        return internals
