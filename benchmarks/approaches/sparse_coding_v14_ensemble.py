"""Ensemble energy-based sparse coding with population voting.

Trains N independent v9 models (different random seeds) and averages their
reconstruction errors at test time. Each model develops slightly different
dictionaries, capturing different aspects of digit structure. Averaging
reduces noise in the reconstruction-error estimates and smooths out
individual model weaknesses.

Architecture: N copies of v9 (incoherence + microsaccade augmentation),
    each trained independently. Classification by averaging reconstruction
    errors across all ensemble members then picking argmin.

Biological analogue:
    - Population coding: the brain doesn't rely on a single cortical column
      to recognize a digit. Multiple columns in IT cortex process the same
      stimulus in parallel, each with slightly different tuning. Decisions
      are based on the population response (Averbeck et al. 2006).
    - Signal-to-noise improvement: averaging N independent noisy estimates
      reduces noise by sqrt(N). This is equivalent to temporal averaging
      across fixations, but implemented spatially across cortical columns.
    - Ensemble diversity = different random seeds = different initial
      synaptic configurations leading to different learned features

Based on:
    - v9 architecture (v6 + microsaccade augmentation)
    - Averbeck et al. (2006): Neural correlations, population coding
    - Georgopoulos et al. (1986): Population vector hypothesis
    - Dietterich (2000): Ensemble methods in machine learning
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


class _SingleModel:
    """A single v9-style model (internal, not an MNISTApproach)."""

    def __init__(
        self,
        n_features_per_class: int,
        n_classes: int,
        n_settle: int,
        sparsity: float,
        infer_rate: float,
        learn_rate: float,
        incoherence_rate: float,
        max_shift: int,
        epochs: int,
        batch_size: int,
        rng: np.random.Generator,
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
        self.rng = rng
        self.dictionaries: list[np.ndarray] = []

    def _settle(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """ISTA settling."""
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
        """Compute per-sample reconstruction error."""
        z = self._settle(x, d)
        recon = z @ d.T
        return np.mean((x - recon) ** 2, axis=1)

    def _apply_incoherence(self, k: int) -> None:
        """Push dictionary k away from other dictionaries."""
        d_k = self.dictionaries[k]
        penalty = np.zeros_like(d_k)
        for j in range(self.n_classes):
            if j == k:
                continue
            d_j = self.dictionaries[j]
            overlap = d_j @ (d_j.T @ d_k)
            penalty += overlap
        self.dictionaries[k] -= self.incoherence_rate * penalty

    def train(self, images: np.ndarray, labels: np.ndarray) -> list[float]:
        """Train and return per-epoch accuracies."""
        n_samples, n_px = images.shape

        self.dictionaries = []
        for _ in range(self.n_classes):
            d = self.rng.normal(0, 1.0, (n_px, self.n_features_per_class))
            norms = np.linalg.norm(d, axis=0, keepdims=True) + 1e-8
            d /= norms
            self.dictionaries.append(d)

        epoch_accs = []
        for epoch in range(self.epochs):
            perm = self.rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_batch = images[idx]
                y_batch = labels[idx]
                x_aug = _random_shift(x_batch, self.max_shift, self.rng)

                for k in range(self.n_classes):
                    mask = y_batch == k
                    if mask.sum() < 2:  # noqa: PLR2004
                        continue

                    x_k = x_aug[mask]
                    bs = x_k.shape[0]
                    z = self._settle(x_k, self.dictionaries[k])
                    residual = x_k - z @ self.dictionaries[k].T
                    self.dictionaries[k] += self.learn_rate * (residual.T @ z) / bs
                    self._apply_incoherence(k)
                    norms = np.linalg.norm(
                        self.dictionaries[k], axis=0, keepdims=True
                    )
                    self.dictionaries[k] /= norms + 1e-8

            # Quick eval
            eval_idx = self.rng.choice(
                n_samples, size=min(2000, n_samples), replace=False
            )
            preds = self.predict(images[eval_idx])
            acc = float(np.mean(preds == labels[eval_idx]))
            epoch_accs.append(acc)

        return epoch_accs

    def get_errors(self, images: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors against all dictionaries.

        Args:
            images: Input images, shape (N, 784).

        Returns:
            Errors, shape (N, n_classes).
        """
        n = images.shape[0]
        errors = np.zeros((n, self.n_classes), dtype=np.float64)
        batch_sz = 1000
        for start in range(0, n, batch_sz):
            x_batch = images[start : start + batch_sz]
            bs = x_batch.shape[0]
            for k in range(self.n_classes):
                err = self._recon_error(x_batch, self.dictionaries[k])
                errors[start : start + bs, k] = err
        return errors

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Classify by competitive reconstruction."""
        errors = self.get_errors(images)
        return np.argmin(errors, axis=1).astype(np.uint8)


class SparseCodingV14Ensemble(MNISTApproach):
    """Ensemble of v9 models with population voting.

    Trains N independent models and averages reconstruction errors at
    test time. Each model has a different random seed, leading to
    different learned dictionaries and complementary error patterns.

    Args:
        n_models: Number of ensemble members.
        n_features_per_class: Dictionary atoms per class per model.
        n_classes: Number of digit classes.
        n_settle: ISTA iterations per settling run.
        sparsity: Soft-threshold lambda.
        infer_rate: ISTA step size.
        learn_rate: Dictionary learning rate.
        incoherence_rate: Inter-dictionary repulsion strength.
        max_shift: Maximum augmentation shift in pixels.
        epochs: Training epochs per model.
        batch_size: Mini-batch size.
        seed: Base random seed.
    """

    name = "sparse_coding_v14"
    uses_backprop = False

    def __init__(
        self,
        n_models: int = 3,
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
        self.n_models = n_models
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
        self.seed = seed

        self.models: list[_SingleModel] = []

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train N independent models sequentially.

        Args:
            images: Training images, shape (N, 784), float64 in [0, 1].
            labels: Training labels, shape (N,), int in [0, 9].
        """
        self.models = []

        for i in range(self.n_models):
            print(f"  --- Ensemble member {i + 1}/{self.n_models} ---")
            rng = np.random.default_rng(self.seed + i * 1000)
            model = _SingleModel(
                n_features_per_class=self.n_features_per_class,
                n_classes=self.n_classes,
                n_settle=self.n_settle,
                sparsity=self.sparsity,
                infer_rate=self.infer_rate,
                learn_rate=self.learn_rate,
                incoherence_rate=self.incoherence_rate,
                max_shift=self.max_shift,
                epochs=self.epochs,
                batch_size=self.batch_size,
                rng=rng,
            )
            epoch_accs = model.train(images, labels)
            self.models.append(model)

            # Log last epoch accuracy for this member
            final_acc = epoch_accs[-1] if epoch_accs else 0.0
            print(f"    Member {i + 1} final train acc: {final_acc:.4f}")

            # Track overall history (use average of individual model accs)
            if i == 0:
                for ep_idx, acc in enumerate(epoch_accs):
                    self.history.append(
                        EpochMetrics(
                            epoch=ep_idx + 1, train_acc=acc, loss=0.0
                        )
                    )

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Classify by averaging reconstruction errors across ensemble.

        For each test image, compute reconstruction errors against all
        10 class dictionaries for each ensemble member. Average the errors
        across members, then pick the class with lowest mean error.

        Args:
            images: Test images, shape (N, 784).

        Returns:
            Predicted labels, shape (N,), int in [0, 9].
        """
        if not self.models:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        n = images.shape[0]
        total_errors = np.zeros((n, self.n_classes), dtype=np.float64)

        for model in self.models:
            total_errors += model.get_errors(images)

        avg_errors = total_errors / len(self.models)
        return np.argmin(avg_errors, axis=1).astype(np.uint8)

    def get_internals(self) -> dict[str, object]:
        """Expose ensemble internals.

        Returns:
            Dict with per-model, per-class dictionaries.
        """
        internals: dict[str, object] = {}
        for m_idx, model in enumerate(self.models):
            for k, d in enumerate(model.dictionaries):
                internals[f"model_{m_idx}_dict_{k}"] = d
        return internals
