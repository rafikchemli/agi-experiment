"""Learned event encoder via autoencoder.

Trains a small MLP autoencoder on raw event features (16 dims) to learn
a compressed representation. The bottleneck codes become the input to
the sparse dictionary — no hand-designed features like displacement or
magnitude. If the autoencoder discovers displacement-like features in
its bottleneck, that's evidence the causal structure emerges from data.

Architecture:
    raw (16d) → encoder (16→32→latent_dim) → decoder (latent_dim→32→16) → raw

Training uses MSE reconstruction loss with simple SGD. No backpropagation
through the dictionary — the autoencoder is trained separately (Phase 1),
then dictionary learning runs on the bottleneck codes (Phase 2).
"""

from __future__ import annotations

import numpy as np


class LearnedEncoder:
    """MLP autoencoder for learning event representations.

    Attributes:
        latent_dim: Dimensionality of the bottleneck (encoder output).
        hidden_dim: Width of the hidden layers.
        learn_rate: SGD step size.
    """

    def __init__(
        self,
        input_dim: int = 16,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        learn_rate: float = 0.005,
        seed: int = 42,
    ) -> None:
        """Initialize the autoencoder.

        Args:
            input_dim: Dimensionality of raw event vectors.
            latent_dim: Bottleneck dimensionality.
            hidden_dim: Hidden layer width.
            learn_rate: SGD learning rate.
            seed: Random seed for reproducibility.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learn_rate = learn_rate
        self._rng = np.random.default_rng(seed)

        # Xavier initialization
        self._w1 = self._rng.standard_normal(
            (input_dim, hidden_dim)
        ) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self._b1 = np.zeros(hidden_dim)

        self._w2 = self._rng.standard_normal(
            (hidden_dim, latent_dim)
        ) * np.sqrt(2.0 / (hidden_dim + latent_dim))
        self._b2 = np.zeros(latent_dim)

        self._w3 = self._rng.standard_normal(
            (latent_dim, hidden_dim)
        ) * np.sqrt(2.0 / (latent_dim + hidden_dim))
        self._b3 = np.zeros(hidden_dim)

        self._w4 = self._rng.standard_normal(
            (hidden_dim, input_dim)
        ) * np.sqrt(2.0 / (hidden_dim + input_dim))
        self._b4 = np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode raw features to bottleneck representation.

        Args:
            x: Input data of shape (N, input_dim).

        Returns:
            Bottleneck codes of shape (N, latent_dim).
        """
        h1 = _relu(x @ self._w1 + self._b1)
        return _relu(h1 @ self._w2 + self._b2)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode bottleneck codes back to raw feature space.

        Args:
            z: Bottleneck codes of shape (N, latent_dim).

        Returns:
            Reconstructed features of shape (N, input_dim).
        """
        h3 = _relu(z @ self._w3 + self._b3)
        return h3 @ self._w4 + self._b4

    def train(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> list[dict[str, float | int]]:
        """Train autoencoder on raw event data.

        Args:
            data: Training data of shape (N, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of dicts with keys "epoch" and "loss" per epoch.
        """
        n = data.shape[0]
        history: list[dict[str, float | int]] = []

        for epoch in range(1, epochs + 1):
            idx = self._rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = idx[start : start + batch_size]
                x = data[batch_idx]
                b = x.shape[0]

                # Forward pass
                h1 = _relu(x @ self._w1 + self._b1)
                z = _relu(h1 @ self._w2 + self._b2)
                h3 = _relu(z @ self._w3 + self._b3)
                recon = h3 @ self._w4 + self._b4

                # Reconstruction loss
                residual = recon - x
                loss = float(np.mean(residual**2))

                # Backward pass (manual gradients)
                d_recon = 2.0 * residual / (b * self.input_dim)

                # Layer 4: linear output
                d_w4 = h3.T @ d_recon
                d_b4 = d_recon.sum(axis=0)
                d_h3 = d_recon @ self._w4.T

                # Layer 3: ReLU
                d_h3 = d_h3 * (z @ self._w3 + self._b3 > 0).astype(float)
                d_w3 = z.T @ d_h3
                d_b3 = d_h3.sum(axis=0)
                d_z = d_h3 @ self._w3.T

                # Layer 2: ReLU
                d_z = d_z * (h1 @ self._w2 + self._b2 > 0).astype(float)
                d_w2 = h1.T @ d_z
                d_b2 = d_z.sum(axis=0)
                d_h1 = d_z @ self._w2.T

                # Layer 1: ReLU
                d_h1 = d_h1 * (x @ self._w1 + self._b1 > 0).astype(float)
                d_w1 = x.T @ d_h1
                d_b1 = d_h1.sum(axis=0)

                # SGD update
                self._w4 -= self.learn_rate * d_w4
                self._b4 -= self.learn_rate * d_b4
                self._w3 -= self.learn_rate * d_w3
                self._b3 -= self.learn_rate * d_b3
                self._w2 -= self.learn_rate * d_w2
                self._b2 -= self.learn_rate * d_b2
                self._w1 -= self.learn_rate * d_w1
                self._b1 -= self.learn_rate * d_b1

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append({"epoch": epoch, "loss": avg_loss})
            if epoch % 20 == 0 or epoch == 1:
                print(f"    AE Epoch {epoch}/{epochs} — loss: {avg_loss:.6f}")

        return history


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0.0, x)
