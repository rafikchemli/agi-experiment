"""Tests for the Forward-Forward algorithm implementation."""

import numpy as np
import pytest

from benchmarks.approaches.forward_forward import (
    ForwardForward,
    _FFLayer,
    _overlay_labels,
    _random_wrong_labels,
)
from benchmarks.mnist_loader import MNISTData, split_validation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def fake_images(rng: np.random.Generator) -> np.ndarray:
    """Small batch of fake 784-pixel images."""
    return rng.random((100, 784))


@pytest.fixture
def fake_labels(rng: np.random.Generator) -> np.ndarray:
    """Labels matching fake_images."""
    return rng.integers(0, 10, size=100).astype(np.uint8)


# ---------------------------------------------------------------------------
# Label overlay
# ---------------------------------------------------------------------------


class TestOverlayLabels:
    """Tests for the label overlay function."""

    def test_first_10_pixels_contain_one_hot(
        self, fake_images: np.ndarray, fake_labels: np.ndarray
    ) -> None:
        """Correct label pixel is set, others in first 10 are zero."""
        result = _overlay_labels(fake_images, fake_labels)
        for i in range(len(fake_labels)):
            label = fake_labels[i]
            # The label pixel should be non-zero
            assert result[i, label] > 0
            # Other pixels in the first 10 should be zero
            for j in range(10):
                if j != label:
                    assert result[i, j] == 0.0

    def test_pixels_after_10_unchanged(
        self, fake_images: np.ndarray, fake_labels: np.ndarray
    ) -> None:
        """Pixels 10-783 are not modified."""
        result = _overlay_labels(fake_images, fake_labels)
        np.testing.assert_array_equal(result[:, 10:], fake_images[:, 10:])

    def test_overlay_shape_preserved(
        self, fake_images: np.ndarray, fake_labels: np.ndarray
    ) -> None:
        """Output shape matches input."""
        result = _overlay_labels(fake_images, fake_labels)
        assert result.shape == fake_images.shape


# ---------------------------------------------------------------------------
# Random wrong labels
# ---------------------------------------------------------------------------


class TestRandomWrongLabels:
    """Tests for negative sample label generation."""

    def test_never_matches_correct(
        self, fake_labels: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Wrong labels must differ from correct labels."""
        wrong = _random_wrong_labels(fake_labels, 10, rng)
        assert np.all(wrong != fake_labels)

    def test_in_valid_range(self, fake_labels: np.ndarray, rng: np.random.Generator) -> None:
        """Wrong labels are in [0, 9]."""
        wrong = _random_wrong_labels(fake_labels, 10, rng)
        assert np.all(wrong >= 0)
        assert np.all(wrong < 10)


# ---------------------------------------------------------------------------
# FFLayer
# ---------------------------------------------------------------------------


class TestFFLayer:
    """Tests for a single Forward-Forward layer."""

    def test_forward_output_shape(self, rng: np.random.Generator) -> None:
        """Forward pass produces correct output shape."""
        layer = _FFLayer(784, 500, threshold=2.0, lr=0.03, rng=rng)
        x = rng.random((32, 784))
        out = layer.forward(x)
        assert out.shape == (32, 500)

    def test_forward_nonnegative(self, rng: np.random.Generator) -> None:
        """ReLU ensures all outputs are non-negative."""
        layer = _FFLayer(784, 500, threshold=2.0, lr=0.03, rng=rng)
        x = rng.random((32, 784))
        out = layer.forward(x)
        assert np.all(out >= 0)

    def test_goodness_shape(self, rng: np.random.Generator) -> None:
        """Goodness is a scalar per sample."""
        layer = _FFLayer(784, 500, threshold=2.0, lr=0.03, rng=rng)
        x = rng.random((32, 784))
        acts = layer.forward(x)
        g = layer.goodness(acts)
        assert g.shape == (32,)

    def test_train_step_reduces_loss(self, rng: np.random.Generator) -> None:
        """A few training steps should reduce the loss."""
        layer = _FFLayer(784, 500, threshold=2.0, lr=0.03, rng=rng)
        x = rng.random((64, 784))
        labels = rng.integers(0, 10, size=64).astype(np.uint8)
        wrong = _random_wrong_labels(labels, 10, rng)

        x_pos = _overlay_labels(x, labels)
        x_neg = _overlay_labels(x, wrong)

        losses = []
        for _ in range(20):
            loss, _, _ = layer.train_step(x_pos, x_neg)
            losses.append(loss)

        # Loss should decrease over training steps
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# ForwardForward end-to-end
# ---------------------------------------------------------------------------


class TestForwardForward:
    """Integration tests for the full Forward-Forward approach."""

    def test_predict_shape(self, fake_images: np.ndarray, fake_labels: np.ndarray) -> None:
        """Predictions have correct shape and range."""
        ff = ForwardForward(hidden_sizes=[32, 32], epochs=5, batch_size=50, log_every=100)
        ff.train(fake_images, fake_labels)
        preds = ff.predict(fake_images)
        assert preds.shape == (100,)
        assert np.all(preds >= 0)
        assert np.all(preds <= 9)

    def test_predict_raises_before_train(self, fake_images: np.ndarray) -> None:
        """Predict raises if model hasn't been trained."""
        ff = ForwardForward()
        with pytest.raises(RuntimeError, match="not trained"):
            ff.predict(fake_images)

    def test_trains_above_chance(self) -> None:
        """On a simple synthetic task, FF should beat random chance (10%)."""
        rng = np.random.default_rng(42)

        # Create simple synthetic data: each class has a distinct pattern
        n_per_class = 50
        images = np.zeros((n_per_class * 10, 784), dtype=np.float64)
        labels = np.zeros(n_per_class * 10, dtype=np.uint8)

        for digit in range(10):
            start = digit * n_per_class
            end = start + n_per_class
            # Each digit activates a different set of ~78 pixels
            base = digit * 78
            images[start:end, base : base + 78] = 0.8 + rng.random((n_per_class, 78)) * 0.2
            # Add small noise everywhere
            images[start:end] += rng.random((n_per_class, 784)) * 0.05
            labels[start:end] = digit

        ff = ForwardForward(
            hidden_sizes=[64, 64],
            epochs=50,
            batch_size=100,
            log_every=100,  # suppress output
        )
        ff.train(images, labels)
        preds = ff.predict(images)
        accuracy = float(np.mean(preds == labels))

        # Should significantly beat chance (10%)
        assert accuracy > 0.4, f"Accuracy {accuracy:.1%} is too close to chance"

    def test_get_internals(self, fake_images: np.ndarray, fake_labels: np.ndarray) -> None:
        """get_internals returns weight matrices after training."""
        ff = ForwardForward(hidden_sizes=[32, 32], epochs=2, batch_size=50, log_every=100)
        ff.train(fake_images, fake_labels)
        internals = ff.get_internals()
        assert "w1" in internals
        assert "w2" in internals
        assert internals["w1"].shape == (784, 32)  # type: ignore[union-attr]
        assert internals["w2"].shape == (32, 32)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Validation split
# ---------------------------------------------------------------------------


class TestSplitValidation:
    """Tests for the MNIST validation split utility."""

    def test_split_sizes(self) -> None:
        """Split produces correct train/val sizes."""
        data = MNISTData(
            train_images=np.zeros((1000, 784)),
            train_labels=np.zeros(1000, dtype=np.uint8),
            test_images=np.zeros((200, 784)),
            test_labels=np.zeros(200, dtype=np.uint8),
        )
        split = split_validation(data, val_size=200)
        assert split.train_images.shape[0] == 800
        assert split.val_images is not None
        assert split.val_images.shape[0] == 200
        assert split.test_images.shape[0] == 200

    def test_split_deterministic(self) -> None:
        """Same seed produces same split."""
        data = MNISTData(
            train_images=np.arange(500 * 784, dtype=np.float64).reshape(500, 784),
            train_labels=np.arange(500, dtype=np.uint8) % 10,
            test_images=np.zeros((100, 784)),
            test_labels=np.zeros(100, dtype=np.uint8),
        )
        s1 = split_validation(data, val_size=100, seed=123)
        s2 = split_validation(data, val_size=100, seed=123)
        np.testing.assert_array_equal(s1.train_images, s2.train_images)
        np.testing.assert_array_equal(s1.val_images, s2.val_images)
