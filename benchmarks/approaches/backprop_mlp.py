"""Backpropagation MLP baseline — the control group.

A simple feedforward neural network trained with gradient descent.
Pure numpy, no frameworks. This is what we're comparing against.

Architecture: 784 -> 300 (ReLU) -> 10 (softmax)
Learning: SGD with backpropagation, cross-entropy loss, step LR decay
Expected accuracy: ~98% on full MNIST

Adapted from the NumPy official tutorial:
https://numpy.org/numpy-tutorials/tutorial-deep-learning-on-mnist/
"""

import numpy as np

from benchmarks.base import EpochMetrics, MNISTApproach


class BackpropMLP(MNISTApproach):
    """Standard MLP with backpropagation.

    This is the control group. It proves the task is solvable and
    sets the accuracy bar. Every non-backprop approach is compared
    against this baseline.

    Args:
        hidden_size: Number of neurons in the hidden layer.
        learning_rate: Initial SGD learning rate.
        epochs: Number of training passes over the full dataset.
        batch_size: Mini-batch size for SGD.
        lr_decay_every: Halve learning rate every N epochs.
        seed: Random seed for reproducibility.
    """

    name = "backprop_mlp"
    uses_backprop = True

    def __init__(
        self,
        hidden_size: int = 300,
        learning_rate: float = 0.1,
        epochs: int = 15,
        batch_size: int = 64,
        lr_decay_every: int = 3,
        seed: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.lr_init = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_decay_every = lr_decay_every
        self.rng = np.random.default_rng(seed)

        self._val_images: np.ndarray | None = None
        self._val_labels: np.ndarray | None = None

        # Weights initialized during train()
        self.w1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.b2: np.ndarray | None = None

    def set_validation(
        self, images: np.ndarray, labels: np.ndarray
    ) -> None:
        """Set validation data for epoch-level accuracy reporting.

        Args:
            images: Validation images, shape (M, 784).
            labels: Validation labels, shape (M,).
        """
        self._val_images = images
        self._val_labels = labels

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_deriv(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative."""
        return (x > 0).astype(np.float64)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.

        Args:
            x: Input array, shape (batch, 10).

        Returns:
            Probability distribution over classes, shape (batch, 10).
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _one_hot(self, labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
        """One-hot encode labels.

        Args:
            labels: Integer labels, shape (N,).
            n_classes: Number of classes.

        Returns:
            One-hot encoded, shape (N, n_classes).
        """
        oh = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
        oh[np.arange(labels.shape[0]), labels] = 1.0
        return oh

    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train the MLP with mini-batch SGD, backpropagation, and LR decay.

        Args:
            images: Training images, shape (N, 784), float64 in [0, 1].
            labels: Training labels, shape (N,), int in [0, 9].
        """
        n_samples, n_inputs = images.shape
        n_classes = 10

        # Xavier initialization
        scale1 = np.sqrt(2.0 / n_inputs)
        scale2 = np.sqrt(2.0 / self.hidden_size)
        self.w1 = self.rng.normal(0, scale1, (n_inputs, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = self.rng.normal(0, scale2, (self.hidden_size, n_classes))
        self.b2 = np.zeros(n_classes)

        targets = self._one_hot(labels, n_classes)
        lr = self.lr_init

        for epoch in range(self.epochs):
            # Step LR decay
            if epoch > 0 and epoch % self.lr_decay_every == 0:
                lr *= 0.5

            # Shuffle
            perm = self.rng.permutation(n_samples)
            epoch_loss = 0.0
            correct = 0

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x_batch = images[idx]
                y_batch = targets[idx]
                batch_n = len(idx)

                # Forward pass
                z1 = x_batch @ self.w1 + self.b1
                a1 = self._relu(z1)
                z2 = a1 @ self.w2 + self.b2
                a2 = self._softmax(z2)

                # Loss (cross-entropy)
                epoch_loss += -np.sum(y_batch * np.log(a2 + 1e-10))
                correct += np.sum(np.argmax(a2, axis=1) == np.argmax(y_batch, axis=1))

                # Backward pass
                # Softmax + cross-entropy gradient simplifies to (prediction - target)
                dz2 = (a2 - y_batch) / batch_n
                dw2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0)

                da1 = dz2 @ self.w2.T
                dz1 = da1 * self._relu_deriv(z1)
                dw1 = x_batch.T @ dz1
                db1 = np.sum(dz1, axis=0)

                # Update weights
                self.w1 -= lr * dw1
                self.b1 -= lr * db1
                self.w2 -= lr * dw2
                self.b2 -= lr * db2

            acc = correct / n_samples
            avg_loss = epoch_loss / n_samples

            # Validation accuracy
            val_acc: float | None = None
            if self._val_images is not None and self._val_labels is not None:
                val_preds = self.predict(self._val_images)
                val_acc = float(np.mean(val_preds == self._val_labels))

            # Record history
            self.history.append(
                EpochMetrics(epoch=epoch + 1, train_acc=acc, loss=avg_loss, val_acc=val_acc)
            )

            # Log
            log = f"    Epoch {epoch+1}/{self.epochs} — lr: {lr:.4f}, loss: {avg_loss:.4f}, train: {acc:.4f}"
            if val_acc is not None:
                log += f", val: {val_acc:.4f}"
            print(log)

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict digit labels.

        Args:
            images: Test images, shape (N, 784).

        Returns:
            Predicted labels, shape (N,).
        """
        if self.w1 is None or self.w2 is None:
            msg = "Model not trained yet"
            raise RuntimeError(msg)

        z1 = images @ self.w1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.w2 + self.b2
        return np.argmax(z2, axis=1).astype(np.uint8)

    def get_internals(self) -> dict[str, object]:
        """Expose weights for visualization.

        Returns:
            Dict with weight matrices.
        """
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
