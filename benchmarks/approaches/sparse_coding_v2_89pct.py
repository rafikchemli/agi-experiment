"""Sparse predictive coding — the brain's visual processing algorithm.

V1 simple cells learn a dictionary of visual features from natural images.
Each image is encoded as a SPARSE combination of these features — most
neurons are silent, only a few fire. This sparsity is enforced by lateral
inhibition (soft thresholding) during iterative settling (ISTA).

Architecture:
    Dictionary D ∈ R^(784 × n_features)  — visual feature templates (V1 cells)
    Sparse code z ∈ R^n_features          — which features are active (firing rates)
    Readout W ∈ R^(n_features × 10)       — decision layer (association cortex)

    x ≈ D @ z   →   "the image is a weighted sum of active features"

Inference (settling):
    ISTA: z ← max(0, z + α · Dᵀ(x − Dz) − λα)
    Bio: z = firing rates, Dᵀ residual = feedforward drive,
         −λα = lateral inhibition (winner-take-some), max(0,·) = threshold

Learning (Hebbian):
    Dictionary: ΔD ∝ (x − Dz) zᵀ  — error × pre-synaptic activity
    Readout:    ΔW ∝ z (y − ŷ)ᵀ   — activity × reward error
    Bio: both are standard Hebbian rules using only local signals

Uncertainty:
    ||x − Dz||² = reconstruction error after settling
    High error → "I can't explain this input" → uncertain
    Low error → "My features account for this" → confident

Based on:
- Olshausen & Field (1996): Emergence of simple-cell receptive field
  properties by learning a sparse code for natural images
- Rozell et al. (2008): Sparse coding via thresholding and local competition
  (Locally Competitive Algorithm — neural implementation of ISTA)
- Hoyer (2002): Non-negative sparse coding

Biological analogues:
- Dictionary columns = V1 simple cell receptive fields
- ISTA settling = ~100ms cortical recurrent dynamics
- Soft threshold = lateral inhibition via inhibitory interneurons
- Column normalization = homeostatic synaptic scaling
- Non-negative codes = firing rates (neurons can't fire negatively)
- Readout learning = reward-modulated Hebbian plasticity
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


class SparseCodingV2(MNISTApproach):
    """Sparse predictive coding network for MNIST classification (v2 snapshot).

    Phase 1 (every batch): Learn dictionary D of visual features by
    minimizing reconstruction error with sparsity penalty.
    Phase 2 (every batch): Learn readout W from sparse codes to classes.

    Both phases use purely local Hebbian rules — no backpropagation.

    This is the stabilized v2 implementation (ISTA + dictionary learning)
    that achieved 89.1% — preserved as a historical benchmark.

    Args:
        n_features: Number of dictionary atoms (V1 simple cells).
        n_settle: ISTA iterations per image (cortical settling time).
        sparsity: Soft-threshold level λ (lateral inhibition strength).
        infer_rate: ISTA step size (neural integration rate).
        learn_rate: Dictionary learning rate (synaptic plasticity speed).
        sup_rate: Readout learning rate (association learning speed).
        epochs: Number of passes over training data.
        batch_size: Mini-batch size.
        seed: Random seed for reproducibility.
    """

    name = "sparse_coding_v2"
    uses_backprop = False

    def __init__(
        self,
        n_features: int = 500,
        n_settle: int = 75,
        sparsity: float = 0.3,
        infer_rate: float = 0.1,
        learn_rate: float = 0.002,
        sup_rate: float = 0.05,
        epochs: int = 25,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_settle = n_settle
        self.sparsity = sparsity
        self.infer_rate = infer_rate
        self.learn_rate = learn_rate
        self.sup_rate = sup_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # Learned during train()
        self.D: np.ndarray | None = None  # dictionary (784, n_features)
        self.W: np.ndarray | None = None  # readout  (n_features, 10)
        self.b_out: np.ndarray | None = None  # readout bias (10,)

        self._last_uncertainty: np.ndarray | None = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.

        Args:
            x: Logits, shape (B, C).

        Returns:
            Probabilities, shape (B, C).
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _settle(self, x: np.ndarray) -> np.ndarray:
        """ISTA settling — infer sparse codes that explain the input.

        Finds z ≥ 0 that approximately minimizes:
            ½||x − Dz||² + λ||z||₁

        Biologically: iterative recurrent dynamics where feedforward
        drive (Dᵀ residual) competes with lateral inhibition (threshold)
        until a sparse stable representation emerges.

        Args:
            x: Input images, shape (B, 784).

        Returns:
            Sparse codes z, shape (B, n_features), non-negative.
        """
        assert self.D is not None
        b = x.shape[0]
        z = np.zeros((b, self.n_features), dtype=np.float64)
        step = self.infer_rate
        threshold = self.sparsity * step

        for _ in range(self.n_settle):
            # Prediction error: what the current code can't explain
            residual = x - z @ self.D.T  # (B, 784)

            # Feedforward drive: how much does each feature match the residual?
            drive = residual @ self.D  # (B, n_features) = Dᵀ @ residual

            # Gradient step: move codes toward explaining more of the input
            z = z + step * drive

            # Proximal step: soft threshold + non-negativity
            # This is lateral inhibition — only strongly driven features survive
            z = np.maximum(0.0, z - threshold)

            # Hard cap on firing rate — prevents runaway activation
            np.minimum(z, 5.0, out=z)

        return z

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train dictionary and readout with local Hebbian rules.

        For each mini-batch:
        1. SETTLE: run ISTA to find sparse codes for current images
        2. LEARN DICTIONARY: ΔD ∝ residual × codes (Hebbian)
        3. NORMALIZE: keep dictionary columns unit-norm (homeostatic)
        4. LEARN READOUT: ΔW ∝ codes × label_error (reward-modulated Hebbian)

        Args:
            images: Training images, shape (N, 784), float64 in [0, 1].
            labels: Training labels, shape (N,), int in [0, 9].
        """
        n_samples, n_px = images.shape

        # Initialize dictionary with random directions, normalized columns
        self.D = self.rng.normal(0, 1.0, (n_px, self.n_features))
        norms = np.linalg.norm(self.D, axis=0, keepdims=True) + 1e-8
        self.D /= norms

        # Initialize readout (zero — no prior bias toward any class)
        self.W = np.zeros((self.n_features, 10), dtype=np.float64)
        self.b_out = np.zeros(10, dtype=np.float64)

        # One-hot encode labels
        targets = np.zeros((n_samples, 10), dtype=np.float64)
        targets[np.arange(n_samples), labels] = 1.0

        for epoch in range(self.epochs):
            perm = self.rng.permutation(n_samples)
            total_recon = 0.0
            total_sparsity = 0.0
            correct = 0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_batch = images[idx]
                y_batch = targets[idx]
                bs = len(idx)

                # Phase 1: SETTLE — find sparse codes via ISTA
                z = self._settle(x_batch)

                # Reconstruction and sparsity diagnostics
                recon = z @ self.D.T
                residual = x_batch - recon
                total_recon += float(np.sum(residual**2)) / n_px
                total_sparsity += float(np.mean(z > 0))
                n_batches += 1

                # Phase 2: LEARN DICTIONARY — Hebbian: error × activity
                # ΔD = η · residualᵀ @ z / bs
                # "Adjust each feature template toward reducing its prediction error,
                #  weighted by how active that feature was"
                self.D += self.learn_rate * (residual.T @ z) / bs

                # Homeostatic scaling: normalize columns to unit L2 norm
                # Prevents any one feature from dominating
                norms = np.linalg.norm(self.D, axis=0, keepdims=True) + 1e-8
                self.D /= norms

                # Phase 3: LEARN READOUT — reward-modulated Hebbian
                logits = z @ self.W + self.b_out
                y_pred = self._softmax(logits)
                label_error = y_batch - y_pred  # "reward prediction error"

                # ΔW = η · zᵀ @ error / bs
                # "Strengthen connection from active features to correct class"
                self.W += self.sup_rate * (z.T @ label_error) / bs
                self.b_out += self.sup_rate * np.mean(label_error, axis=0)

                # Track accuracy
                preds = np.argmax(logits, axis=1)
                correct += int(np.sum(preds == labels[idx]))

            acc = correct / n_samples
            avg_recon = total_recon / n_samples
            avg_sparse = total_sparsity / n_batches

            self.history.append(
                EpochMetrics(epoch=epoch + 1, train_acc=acc, loss=avg_recon)
            )

            print(
                f"    Epoch {epoch+1}/{self.epochs} — "
                f"recon: {avg_recon:.2f}, "
                f"sparsity: {avg_sparse:.1%} active, "
                f"train acc: {acc:.4f}"
            )

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict by sparse coding + readout, no supervision needed.

        The dictionary encodes what the network "knows" about visual
        structure. Sparse codes capture which features are present.
        The readout maps features to class predictions.

        Uncertainty = reconstruction error: high error means the
        network's generative model can't explain this input well.

        Args:
            images: Test images, shape (N, 784).

        Returns:
            Predicted labels, shape (N,), int in [0, 9].
        """
        if self.D is None:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        all_preds: list[np.ndarray] = []
        all_uncertainty: list[np.ndarray] = []

        for start in range(0, images.shape[0], self.batch_size):
            x_batch = images[start : start + self.batch_size]

            # Settle sparse codes (same ISTA as training, no labels needed)
            z = self._settle(x_batch)

            # Classify from sparse representation
            logits = z @ self.W + self.b_out
            preds = np.argmax(logits, axis=1)
            all_preds.append(preds)

            # Uncertainty = reconstruction error per sample
            recon_error = np.mean((x_batch - z @ self.D.T) ** 2, axis=1)
            all_uncertainty.append(recon_error)

        self._last_uncertainty = np.concatenate(all_uncertainty)
        return np.concatenate(all_preds).astype(np.uint8)

    def get_internals(self) -> dict[str, object]:
        """Expose internal state for analysis.

        Returns:
            Dict with dictionary, readout weights, and uncertainty.
        """
        internals: dict[str, object] = {
            "dictionary": self.D,
            "readout_W": self.W,
            "readout_b": self.b_out,
        }
        if self._last_uncertainty is not None:
            internals["uncertainty"] = self._last_uncertainty
            internals["mean_uncertainty"] = float(np.mean(self._last_uncertainty))
            internals["std_uncertainty"] = float(np.std(self._last_uncertainty))
        return internals
