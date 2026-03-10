"""Sparse dictionary learning via ISTA settling and Hebbian updates.

Learns an overcomplete dictionary of basis atoms from event vectors using
iterative shrinkage-thresholding (ISTA) for sparse inference and a local
Hebbian rule for weight updates. No backpropagation required.
"""

from __future__ import annotations

import numpy as np


class SparseDictionary:
    """Sparse dictionary learning via ISTA settling + Hebbian updates.

    Attributes:
        n_atoms: Number of dictionary atoms.
        n_settle: Number of ISTA settling iterations per inference.
        sparsity: Sparsity penalty (soft-threshold level).
        infer_rate: Step size for ISTA inference.
        learn_rate: Step size for Hebbian dictionary updates.
    """

    def __init__(
        self,
        n_atoms: int = 30,
        n_settle: int = 50,
        sparsity: float = 0.05,
        infer_rate: float = 0.1,
        learn_rate: float = 0.02,
        seed: int = 42,
    ) -> None:
        """Initialize the sparse dictionary learner.

        Args:
            n_atoms: Number of dictionary atoms (columns of D).
            n_settle: Number of ISTA settling iterations.
            sparsity: Sparsity penalty coefficient.
            infer_rate: Step size for sparse code inference.
            learn_rate: Step size for Hebbian dictionary update.
            seed: Random seed for reproducibility.
        """
        self.n_atoms = n_atoms
        self.n_settle = n_settle
        self.sparsity = sparsity
        self.infer_rate = infer_rate
        self.learn_rate = learn_rate
        self._rng = np.random.default_rng(seed)
        self._D: np.ndarray | None = None

    def train(
        self,
        data: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
    ) -> list[dict[str, float | int]]:
        """Train dictionary on data.

        Args:
            data: Training data of shape (N, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size for stochastic updates.

        Returns:
            List of dicts with keys "epoch" and "loss" per epoch.
        """
        n_samples, input_dim = data.shape

        # Initialize D as random normalized columns: (input_dim, n_atoms)
        self._D = self._rng.standard_normal((input_dim, self.n_atoms))
        norms = np.linalg.norm(self._D, axis=0, keepdims=True) + 1e-8
        self._D /= norms

        history: list[dict[str, float | int]] = []

        for epoch in range(1, epochs + 1):
            # Shuffle data each epoch
            indices = self._rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                x = data[batch_idx]
                b = x.shape[0]

                # ISTA settle to get sparse codes z
                z = self._ista_settle(x)

                # Hebbian update
                residual = x - z @ self._D.T
                self._D += self.learn_rate * (residual.T @ z) / b

                # Normalize columns
                col_norms = (
                    np.linalg.norm(self._D, axis=0, keepdims=True) + 1e-8
                )
                self._D /= col_norms

                # Accumulate loss (mean reconstruction error)
                batch_loss = float(np.mean(residual**2))
                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append({"epoch": epoch, "loss": avg_loss})
            print(
                f"    Epoch {epoch}/{epochs}"
                f" — loss: {avg_loss:.6f}"
            )

        return history

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Infer sparse codes via ISTA settling.

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Sparse codes of shape (N, n_atoms).

        Raises:
            RuntimeError: If the dictionary has not been trained yet.
        """
        if self._D is None:
            msg = "Dictionary not trained yet. Call train() first."
            raise RuntimeError(msg)
        return self._ista_settle(data)

    def reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error.

        Args:
            data: Input data of shape (N, input_dim).

        Returns:
            Per-sample MSE of shape (N,).

        Raises:
            RuntimeError: If the dictionary has not been trained yet.
        """
        if self._D is None:
            msg = "Dictionary not trained yet. Call train() first."
            raise RuntimeError(msg)
        z = self._ista_settle(data)
        residual = data - z @ self._D.T
        return np.mean(residual**2, axis=1)

    @property
    def dictionary(self) -> np.ndarray:
        """The dictionary matrix D of shape (input_dim, n_atoms).

        Raises:
            RuntimeError: If the dictionary has not been trained yet.
        """
        if self._D is None:
            msg = "Dictionary not trained yet. Call train() first."
            raise RuntimeError(msg)
        return self._D

    def _ista_settle(self, x: np.ndarray) -> np.ndarray:
        """Run ISTA settling to infer sparse codes.

        Args:
            x: Input batch of shape (batch, input_dim).

        Returns:
            Sparse codes z of shape (batch, n_atoms).
        """
        assert self._D is not None  # noqa: S101
        batch = x.shape[0]
        z = np.zeros((batch, self.n_atoms))

        for _ in range(self.n_settle):
            residual = x - z @ self._D.T
            drive = residual @ self._D
            z = z + self.infer_rate * drive
            z = np.maximum(0.0, z - self.sparsity * self.infer_rate)
            np.minimum(z, 5.0, out=z)

        return z
