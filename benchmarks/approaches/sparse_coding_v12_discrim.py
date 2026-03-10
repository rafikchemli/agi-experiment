"""Discriminative dictionary refinement — reconstruction + discrimination.

Builds on v9 (augmentation + incoherence) and adds a discriminative
refinement phase. After the standard reconstruction-based training,
a second phase adjusts dictionaries to not just explain their own class
well, but to explain other classes POORLY.

The key insight: v9's dictionaries are good at reconstruction but not at
discrimination. Dictionary_3 might explain digit 5 almost as well as
digit 3. Incoherence helps (v6/v9 push dictionary subspaces apart), but
it doesn't use the actual data. Discriminative refinement does.

Architecture: Same as v9 — 10 class-specific dictionaries with incoherence
    regularization and microsaccade augmentation.

Training phases:
    Phase 1 (epochs 1-25): Standard v9 training (reconstruction + incoherence)
    Phase 2 (epochs 26-35): Add discriminative refinement:
        For each batch, for each class k:
        1. Standard reconstruction update: ΔD_k += lr * (residual_k × z_k)
        2. Discriminative penalty: for wrong class j ≠ k samples that D_k
           explains well (low reconstruction error), push D_k AWAY from
           explaining them: ΔD_k -= disc_rate * (x_j - D_k @ z_j_k) × z_j_k
           This increases reconstruction error for wrong-class images.

Biological analogue:
    - Phase 1 = developmental learning: the visual cortex learns to represent
      its inputs efficiently (unsupervised, first years of life)
    - Phase 2 = task-dependent refinement: when the organism needs to
      discriminate (e.g., predator vs prey), top-down attention from
      prefrontal cortex modulates V1/IT representations to sharpen
      category boundaries (Li et al. 2004, Freedman & Assad 2006)
    - The discriminative signal is like dopaminergic feedback — it doesn't
      teach the neurons directly, but modulates plasticity based on
      prediction errors at the decision level

Based on:
    - v9 architecture
    - Mairal et al. (2009): Supervised dictionary learning
    - Jiang et al. (2011): Learning discriminative dictionary for group
      sparse representation
    - Li et al. (2004): Perceptual learning and top-down attention in V1
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


class SparseCodingV12Discrim(MNISTApproach):
    """Discriminative dictionary refinement on top of v9.

    Two-phase training: reconstruction-first, then discriminative refinement
    that penalizes dictionaries for explaining wrong-class images.

    Args:
        n_features_per_class: Dictionary atoms per class.
        n_classes: Number of digit classes.
        n_settle: ISTA iterations per settling run.
        sparsity: Soft-threshold lambda.
        infer_rate: ISTA step size.
        learn_rate: Dictionary learning rate.
        incoherence_rate: Inter-dictionary repulsion strength.
        disc_rate: Discriminative refinement rate.
        disc_start_epoch: Epoch to start discriminative refinement.
        max_shift: Maximum augmentation shift in pixels.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        seed: Random seed.
    """

    name = "sparse_coding_v12"
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
        disc_rate: float = 0.003,
        disc_start_epoch: int = 20,
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
        self.disc_rate = disc_rate
        self.disc_start_epoch = disc_start_epoch
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

    def _discriminative_update(
        self, k: int, x_others: np.ndarray
    ) -> None:
        """Push dictionary k away from explaining wrong-class images.

        For wrong-class images that D_k explains well (low recon error),
        adjust D_k to increase their reconstruction error. This is like
        top-down attention saying "don't represent things that aren't
        your category."

        Args:
            k: Class index for this dictionary.
            x_others: Images from OTHER classes, shape (B, 784).
        """
        if x_others.shape[0] == 0:
            return

        d = self.dictionaries[k]
        z = self._settle(x_others, d)
        residual = x_others - z @ d.T
        bs = x_others.shape[0]

        # Anti-Hebbian: decrease ability to reconstruct wrong-class images
        # ΔD -= disc_rate * residual^T @ z (negative of the Hebbian update)
        self.dictionaries[k] -= self.disc_rate * (residual.T @ z) / bs

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train with two phases: reconstruction then discrimination.

        Phase 1 (epochs 1 to disc_start_epoch): Standard v9 training.
        Phase 2 (epochs disc_start_epoch+1 to end): Add discriminative
        refinement that penalizes D_k for explaining non-k images.

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
            do_discrim = epoch >= self.disc_start_epoch

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_batch = images[idx]
                y_batch = labels[idx]

                # Microsaccade augmentation
                x_aug = _random_shift(x_batch, self.max_shift, self.rng)

                for k in range(self.n_classes):
                    mask_k = y_batch == k
                    if mask_k.sum() < 2:  # noqa: PLR2004
                        continue

                    x_k = x_aug[mask_k]
                    bs = x_k.shape[0]
                    d = self.dictionaries[k]

                    # Standard reconstruction update
                    z = self._settle(x_k, d)
                    residual = x_k - z @ d.T
                    total_recon += float(np.sum(residual**2)) / n_px
                    self.dictionaries[k] += self.learn_rate * (residual.T @ z) / bs

                    # Discriminative refinement (phase 2 only)
                    if do_discrim:
                        mask_other = ~mask_k
                        if mask_other.sum() > 0:
                            # Sample a small subset of other-class images
                            other_imgs = x_aug[mask_other]
                            n_sample = min(bs, other_imgs.shape[0])
                            sample_idx = self.rng.choice(
                                other_imgs.shape[0], n_sample, replace=False
                            )
                            self._discriminative_update(k, other_imgs[sample_idx])

                    # Incoherence penalty
                    self._apply_incoherence(k)

                    # Homeostatic normalization
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

            phase = "P2-disc" if do_discrim else "P1-recon"
            self.history.append(
                EpochMetrics(epoch=epoch + 1, train_acc=acc, loss=avg_recon)
            )
            print(
                f"    Epoch {epoch + 1}/{self.epochs} [{phase}] — "
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
