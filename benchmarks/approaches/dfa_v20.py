"""Direct Feedback Alignment — biologically plausible supervised learning.

Standard backpropagation requires the transpose of forward weights to
propagate gradients backward (the "weight transport problem"). DFA solves
this by sending the output error directly to each hidden layer through
fixed random matrices. No backward pass through the network.

Update rule for hidden layer l:
    ΔW_l = -lr * h_{l-1}^T @ (B_l @ e)
    where:
        e = output error (softmax probabilities - one-hot labels)
        B_l = fixed random feedback matrix (output_dim → hidden_l_dim)
        h_{l-1} = pre-activation input to layer l (with ReLU derivative)

The output layer uses standard gradient: ΔW_out = -lr * h_L^T @ e

Key differences from backprop:
    - NO backward pass through the network
    - Feedback matrices B_l are random and FIXED (never updated)
    - Each hidden layer receives the SAME error signal, just projected
      differently — like a broadcast
    - Forward weights can still be learned because random projections
      preserve enough directional information (Lillicrap et al. 2016)

Biological analogue:
    - The weight transport problem: backprop requires each synapse to
      "know" the weight of the synapse in the reverse direction. There's
      no known biological mechanism for this (Grossberg 1987).
    - DFA's random feedback is like neuromodulatory signals (dopamine,
      norepinephrine) that broadcast error information globally. Each
      neuron receives the same signal but responds differently based on
      its random connectivity to the error source.
    - Lillicrap et al. (2016) showed that forward weights learn to align
      with the random feedback, making DFA converge to a useful solution
      even though feedback is random. This "feedback alignment" phenomenon
      is a form of self-organization.

Architecture: 784 → 2000 (ReLU) → 1000 (ReLU) → 10 (softmax)

Based on:
    - Lillicrap et al. (2016): Random synaptic feedback weights support
      error backpropagation for deep learning
    - Nøkland (2016): Direct Feedback Alignment provides learning in
      deep neural networks
    - Refinetti et al. (2021): Align, then memorise: the dynamics of
      learning with feedback alignment
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        logits: Shape (N, C).

    Returns:
        Probabilities, shape (N, C), rows sum to 1.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation.

    Args:
        x: Input array.

    Returns:
        max(0, x).
    """
    return np.maximum(0, x)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    """ReLU derivative (step function).

    Args:
        x: Pre-activation values.

    Returns:
        1.0 where x > 0, else 0.0.
    """
    return (x > 0).astype(np.float64)


class DFAV20(MNISTApproach):
    """Direct Feedback Alignment — no weight transport, no backward pass.

    Three-layer feedforward network trained with DFA: each hidden layer
    receives the output error projected through a fixed random matrix.
    The output layer uses standard softmax cross-entropy gradient.

    Args:
        hidden1: First hidden layer size.
        hidden2: Second hidden layer size.
        learning_rate: Initial learning rate.
        lr_decay: Multiplicative LR decay per epoch.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        seed: Random seed.
    """

    name = "dfa_v20"
    uses_backprop = False

    def __init__(
        self,
        hidden1: int = 2000,
        hidden2: int = 1000,
        learning_rate: float = 0.01,
        lr_decay: float = 0.97,
        epochs: int = 60,
        batch_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.n_classes = 10

        # Weights (initialized in train)
        self._w1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._w2: np.ndarray | None = None
        self._b2: np.ndarray | None = None
        self._w_out: np.ndarray | None = None
        self._b_out: np.ndarray | None = None

        # Fixed random feedback matrices (initialized in train)
        self._B1: np.ndarray | None = None  # output_dim → hidden1
        self._B2: np.ndarray | None = None  # output_dim → hidden2

    def _forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the network.

        Args:
            x: Input, shape (N, 784).

        Returns:
            Tuple of (z1, h1, z2, h2, probs) where z are pre-activations,
            h are post-activations, probs are softmax outputs.
        """
        z1 = x @ self._w1 + self._b1
        h1 = _relu(z1)

        z2 = h1 @ self._w2 + self._b2
        h2 = _relu(z2)

        logits = h2 @ self._w_out + self._b_out
        probs = _softmax(logits)

        return z1, h1, z2, h2, probs

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train with Direct Feedback Alignment.

        Each hidden layer receives the output error projected through a
        fixed random matrix. The output layer uses standard gradient.

        Args:
            images: Training images, shape (N, 784).
            labels: Training labels, shape (N,).
        """
        n_samples, n_px = images.shape

        # Xavier initialization for forward weights
        self._w1 = self.rng.normal(
            0, np.sqrt(2.0 / n_px), (n_px, self.hidden1)
        )
        self._b1 = np.zeros(self.hidden1)
        self._w2 = self.rng.normal(
            0, np.sqrt(2.0 / self.hidden1), (self.hidden1, self.hidden2)
        )
        self._b2 = np.zeros(self.hidden2)
        self._w_out = self.rng.normal(
            0, np.sqrt(2.0 / self.hidden2), (self.hidden2, self.n_classes)
        )
        self._b_out = np.zeros(self.n_classes)

        # Fixed random feedback matrices (DFA core)
        # These are NEVER updated — they provide random projections of
        # the output error to each hidden layer.
        scale1 = np.sqrt(1.0 / self.n_classes)
        scale2 = np.sqrt(1.0 / self.n_classes)
        self._B1 = self.rng.normal(0, scale1, (self.n_classes, self.hidden1))
        self._B2 = self.rng.normal(0, scale2, (self.n_classes, self.hidden2))

        # One-hot encode labels
        one_hot = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        one_hot[np.arange(n_samples), labels] = 1.0

        lr = self.learning_rate

        for epoch in range(self.epochs):
            perm = self.rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_batch = images[idx]
                y_batch = one_hot[idx]
                bs = x_batch.shape[0]

                # Forward
                z1, h1, z2, h2, probs = self._forward(x_batch)

                # Output error: e = probs - one_hot
                e = probs - y_batch  # (bs, 10)

                # Cross-entropy loss
                loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-8), axis=1))
                epoch_loss += loss

                # === DFA updates ===

                # Output layer: standard gradient
                dw_out = h2.T @ e / bs
                db_out = np.mean(e, axis=0)
                self._w_out -= lr * dw_out
                self._b_out -= lr * db_out

                # Hidden layer 2: DFA — error projected through random B2
                # δ2 = (e @ B2) * relu'(z2)
                delta2 = (e @ self._B2) * _relu_derivative(z2)
                dw2 = h1.T @ delta2 / bs
                db2 = np.mean(delta2, axis=0)
                self._w2 -= lr * dw2
                self._b2 -= lr * db2

                # Hidden layer 1: DFA — error projected through random B1
                # δ1 = (e @ B1) * relu'(z1)
                delta1 = (e @ self._B1) * _relu_derivative(z1)
                dw1 = x_batch.T @ delta1 / bs
                db1 = np.mean(delta1, axis=0)
                self._w1 -= lr * dw1
                self._b1 -= lr * db1

                n_batches += 1

            # LR decay
            lr *= self.lr_decay

            # Evaluate
            avg_loss = epoch_loss / n_batches
            eval_idx = self.rng.choice(
                n_samples, size=min(2000, n_samples), replace=False
            )
            preds = self.predict(images[eval_idx])
            acc = float(np.mean(preds == labels[eval_idx]))
            self.history.append(
                EpochMetrics(epoch=epoch + 1, train_acc=acc, loss=avg_loss)
            )
            print(
                f"    Epoch {epoch+1}/{self.epochs} — "
                f"loss: {avg_loss:.4f}, acc: {acc:.4f}, lr: {lr:.5f}"
            )

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict by forward pass and argmax.

        Args:
            images: Test images, shape (N, 784).

        Returns:
            Predicted labels, shape (N,).
        """
        if self._w1 is None:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        _, _, _, _, probs = self._forward(images)
        return np.argmax(probs, axis=1).astype(np.uint8)

    def get_internals(self) -> dict[str, object]:
        """Expose network weights and feedback matrices.

        Returns:
            Dict with forward weights, biases, and fixed feedback matrices.
        """
        return {
            "w1": self._w1,
            "b1": self._b1,
            "w2": self._w2,
            "b2": self._b2,
            "w_out": self._w_out,
            "b_out": self._b_out,
            "B1_feedback": self._B1,
            "B2_feedback": self._B2,
        }
