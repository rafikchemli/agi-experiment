"""Abstract base class for MNIST learning approaches.

Every approach implements train() and predict(). The framework handles
data loading, evaluation, and comparison. Drop a new file in approaches/
to add a new method.
"""

from abc import ABC, abstractmethod

import numpy as np


class EpochMetrics:
    """Metrics recorded at the end of each training epoch.

    Attributes:
        epoch: 1-indexed epoch number.
        train_acc: Training accuracy (0-1).
        val_acc: Validation accuracy if available, else None.
        loss: Training loss.
    """

    __slots__ = ("epoch", "train_acc", "val_acc", "loss")

    def __init__(
        self,
        epoch: int,
        train_acc: float,
        loss: float,
        val_acc: float | None = None,
    ) -> None:
        self.epoch = epoch
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.loss = loss

    def to_dict(self) -> dict[str, float | int | None]:
        """Serialize to a JSON-friendly dict."""
        return {
            "epoch": self.epoch,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "loss": self.loss,
        }


class MNISTApproach(ABC):
    """Interface for an MNIST classification approach.

    Attributes:
        name: Human-readable name shown in comparison tables.
        uses_backprop: Whether this approach uses backpropagation.
        history: Per-epoch metrics populated during train().
    """

    name: str = "unnamed"
    uses_backprop: bool = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

    @property
    def history(self) -> list[EpochMetrics]:
        """Per-epoch training metrics."""
        if not hasattr(self, "_history"):
            self._history: list[EpochMetrics] = []
        return self._history

    @abstractmethod
    def train(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Train on MNIST data.

        Args:
            images: Training images, shape (N, 784), float64 in [0, 1].
            labels: Training labels, shape (N,), int in [0, 9].
        """

    @abstractmethod
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict digit labels for images.

        Args:
            images: Test images, shape (N, 784), float64 in [0, 1].

        Returns:
            Predicted labels, shape (N,), int in [0, 9].
        """

    def get_internals(self) -> dict[str, object]:
        """Expose internal state for analysis.

        Override to return weight matrices, firing rates, uncertainty
        signals, or anything useful for understanding HOW the approach
        works (not just accuracy).

        Returns:
            Dict of named internal state objects.
        """
        return {}
