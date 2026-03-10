"""Hybrid FF + sparse coding — compositional soft fusion.

Uses the actual standalone ForwardForward and SparseCodingV9Augmented classes
(imported, not reimplemented) to ensure each subsystem reaches its standalone
accuracy. Combines their raw score distributions via softmax fusion.

v17 (hard arbitration with reimplemented models): 97.1%
v18 (soft fusion, reimplemented): 96.98% — models degraded due to RNG divergence
v19 (soft fusion, imported standalone models): should hit ~97.3%+

The key insight is that reimplementing models introduces subtle RNG state
differences (evaluation calls consume RNG during training). Using the original
classes guarantees identical training trajectories.

Score-to-probability conversion:
    FF: p_ff = softmax(goodness / T_ff)
    SC: p_sc = softmax(-errors / T_sc)
    Combined: p = w_ff * p_ff + (1 - w_ff) * p_sc

Biological analogue:
    - Compositional multi-stream processing: the brain doesn't have a single
      unified visual system — it has multiple specialized subsystems (V1, V2,
      V4, MT, IT) that develop independently and combine at decision time
      (Ungerleider & Mishkin 1982, Felleman & Van Essen 1991).
    - Each stream has its own local learning rules and representations.
    - Integration happens in prefrontal/parietal cortex via reliability-
      weighted combination (Gold & Shadlen 2007).

Based on:
    - Forward-Forward: Hinton (2022)
    - Sparse coding v9: Olshausen & Field (1996) + incoherence + augmentation
    - Gold & Shadlen (2007): The neural basis of decision making
"""

import numpy as np

from benchmarks.approaches.forward_forward import ForwardForward
from benchmarks.approaches.forward_forward import _overlay_labels
from benchmarks.approaches.sparse_coding_v9_augmented import SparseCodingV9Augmented
from benchmarks.base import EpochMetrics, MNISTApproach


class HybridV19Compose(MNISTApproach):
    """Compositional hybrid using imported standalone FF and SC models.

    Each subsystem is instantiated from its original class, trained
    independently, then combined via softmax probability fusion.

    Args:
        temp_ff: FF softmax temperature.
        temp_sc: SC softmax temperature.
        weight_ff: FF weight in fusion (SC weight = 1 - weight_ff).
        ff_seed: Random seed for FF subsystem.
        sc_seed: Random seed for SC subsystem.
    """

    name = "hybrid_v19"
    uses_backprop = False

    def __init__(
        self,
        temp_ff: float = 1.0,
        temp_sc: float = 0.003,
        weight_ff: float = 0.25,
        ff_seed: int = 42,
        sc_seed: int = 42,
    ) -> None:
        self.temp_ff = temp_ff
        self.temp_sc = temp_sc
        self.weight_ff = weight_ff

        # Instantiate standalone models with their default configs
        self._ff = ForwardForward(seed=ff_seed)
        self._sc = SparseCodingV9Augmented(seed=sc_seed)

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
            for layer in self._ff._layers:
                x = layer.forward(x)
                total_g += layer.goodness(x)
            goodness[:, label] = total_g
        return goodness

    def _sc_errors_all(self, images: np.ndarray) -> np.ndarray:
        """Compute SC reconstruction errors against all dictionaries.

        Args:
            images: Input, shape (N, 784).

        Returns:
            Errors, shape (N, 10).
        """
        n = images.shape[0]
        errors = np.zeros((n, self._sc.n_classes), dtype=np.float64)
        batch_sz = 1000
        for start in range(0, n, batch_sz):
            x_batch = images[start : start + batch_sz]
            bs = x_batch.shape[0]
            for k in range(self._sc.n_classes):
                err = self._sc._recon_error(x_batch, self._sc.dictionaries[k])
                errors[start : start + bs, k] = err
        return errors

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train both subsystems independently.

        Args:
            images: Training images, shape (N, 784).
            labels: Training labels, shape (N,).
        """
        # Train FF (exactly as standalone)
        print("    === Training Forward-Forward (standalone) ===")
        self._ff.train(images, labels)

        # Train SC (exactly as standalone)
        print("    === Training Sparse Coding v9 (standalone) ===")
        self._sc.train(images, labels)

        # Evaluate hybrid
        eval_rng = np.random.default_rng(99)
        eval_idx = eval_rng.choice(
            len(images), size=min(2000, len(images)), replace=False
        )
        preds = self.predict(images[eval_idx])
        acc = float(np.mean(preds == labels[eval_idx]))
        self.history.append(EpochMetrics(epoch=1, train_acc=acc, loss=0.0))
        print(f"    Hybrid train acc: {acc:.4f}")

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict using softmax probability fusion.

        Args:
            images: Test images, shape (N, 784).

        Returns:
            Predicted labels, shape (N,).
        """
        if not self._ff._layers or not self._sc.dictionaries:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        # FF probabilities: softmax(goodness / T_ff)
        goodness = self._ff_goodness_all(images)
        g_scaled = goodness / self.temp_ff
        g_scaled = g_scaled - g_scaled.max(axis=1, keepdims=True)
        exp_g = np.exp(g_scaled)
        p_ff = exp_g / exp_g.sum(axis=1, keepdims=True)

        # SC probabilities: softmax(-errors / T_sc)
        errors = self._sc_errors_all(images)
        e_scaled = -errors / self.temp_sc
        e_scaled = e_scaled - e_scaled.max(axis=1, keepdims=True)
        exp_e = np.exp(e_scaled)
        p_sc = exp_e / exp_e.sum(axis=1, keepdims=True)

        # Weighted fusion
        p_combined = self.weight_ff * p_ff + (1.0 - self.weight_ff) * p_sc

        return np.argmax(p_combined, axis=1).astype(np.uint8)

    def get_internals(self) -> dict[str, object]:
        """Expose internals from both systems.

        Returns:
            Dict with FF weights and SC dictionaries.
        """
        internals: dict[str, object] = {}
        internals.update(self._ff.get_internals())
        internals.update(self._sc.get_internals())
        return internals
