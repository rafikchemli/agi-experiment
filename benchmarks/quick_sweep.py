"""Quick targeted parameter sweep — round 3: sparsity fine-tuning.

Usage:
    uv run python -m benchmarks.quick_sweep
"""

import contextlib
import io
import time

import numpy as np

from benchmarks.approaches.predictive_coding import PredictiveCoding
from benchmarks.mnist_loader import load_mnist, split_validation

CONFIGS: dict[str, dict[str, int | float]] = {
    # Current best (92.7% on full data)
    "sparse_0.05": dict(
        n_features=500, n_settle=50, sparsity=0.05,
        infer_rate=0.1, learn_rate=0.005, sup_rate=0.1,
    ),
    # Even lower sparsity
    "sparse_0.02": dict(
        n_features=500, n_settle=50, sparsity=0.02,
        infer_rate=0.1, learn_rate=0.005, sup_rate=0.1,
    ),
    "sparse_0.01": dict(
        n_features=500, n_settle=50, sparsity=0.01,
        infer_rate=0.1, learn_rate=0.005, sup_rate=0.1,
    ),
    # No sparsity at all (just non-negative)
    "sparse_0.001": dict(
        n_features=500, n_settle=50, sparsity=0.001,
        infer_rate=0.1, learn_rate=0.005, sup_rate=0.1,
    ),
    # Low sparsity + more features
    "700f_sparse_0.02": dict(
        n_features=700, n_settle=50, sparsity=0.02,
        infer_rate=0.1, learn_rate=0.003, sup_rate=0.1,
    ),
    # Low sparsity + 25 settle steps (2x faster)
    "25steps_sparse_0.02": dict(
        n_features=500, n_settle=25, sparsity=0.02,
        infer_rate=0.15, learn_rate=0.005, sup_rate=0.1,
    ),
}


def main() -> None:
    """Run quick sweep round 3."""
    print("Loading MNIST...")
    mnist = load_mnist()
    mnist = split_validation(mnist)

    rng = np.random.default_rng(42)
    n_fast = 5000
    idx = rng.choice(mnist.train_images.shape[0], size=n_fast, replace=False)

    train_images = mnist.train_images[idx]
    train_labels = mnist.train_labels[idx]
    val_images = mnist.val_images[:2000]
    val_labels = mnist.val_labels[:2000]

    print(f"Round 3: sparsity fine-tuning, {n_fast} train, {len(val_labels)} val\n")
    print(f"  {'Config':<25s} {'Val Acc':>8s} {'Time':>6s}")
    print(f"  {'-'*25} {'-'*8} {'-'*6}")

    results: list[tuple[str, float, float]] = []

    for name, cfg in CONFIGS.items():
        model = PredictiveCoding(
            epochs=5, batch_size=512, seed=42, **cfg,  # type: ignore[arg-type]
        )
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            model.train(train_images, train_labels)
            preds = model.predict(val_images)
        elapsed = time.time() - t0

        acc = float(np.mean(preds == val_labels))
        results.append((name, acc, elapsed))
        print(f"  {name:<25s} {acc*100:7.1f}% {elapsed:5.0f}s")

    print(f"\n  Ranking:")
    results.sort(key=lambda x: -x[1])
    for i, (name, acc, elapsed) in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. {name:<25s} {acc*100:5.1f}%  ({elapsed:.0f}s){marker}")


if __name__ == "__main__":
    main()
