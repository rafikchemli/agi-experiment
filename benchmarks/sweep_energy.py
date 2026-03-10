"""Optuna hyperparameter sweep for energy-based sparse coding approach.

Uses a subset of training data and fewer epochs for fast iteration.
Each trial targets ~20-40s. Run 40+ trials to find good hyperparameters.

Usage:
    uv run python -m benchmarks.sweep_energy                    # Run 40 trials
    uv run python -m benchmarks.sweep_energy --n-trials 50      # Run 50 trials
    uv run python -m benchmarks.sweep_energy --full              # Evaluate best on full data
"""

import contextlib
import io
import sys
import time

import numpy as np
import optuna

from benchmarks.approaches.sparse_coding_v5_energy import SparseCodingV5Energy
from benchmarks.mnist_loader import load_mnist, split_validation


def objective(trial: optuna.Trial, data: dict[str, np.ndarray]) -> float:
    """Optuna objective: train on subset, evaluate on validation set.

    Args:
        trial: Optuna trial with hyperparameter suggestions.
        data: Dict with train_images, train_labels, val_images, val_labels.

    Returns:
        Validation accuracy (to maximize).
    """
    model = SparseCodingV5Energy(
        n_features_per_class=trial.suggest_int("n_features_per_class", 50, 400, step=50),
        n_settle=trial.suggest_int("n_settle", 20, 80, step=10),
        sparsity=trial.suggest_float("sparsity", 0.001, 0.1, log=True),
        infer_rate=trial.suggest_float("infer_rate", 0.05, 0.3, log=True),
        learn_rate=trial.suggest_float("learn_rate", 0.001, 0.05, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        epochs=5,
        seed=42,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        model.train(data["train_images"], data["train_labels"])
        preds = model.predict(data["val_images"])

    accuracy = float(np.mean(preds == data["val_labels"]))
    return accuracy


def main() -> None:
    """Run hyperparameter sweep."""
    n_trials = 40
    run_full = False

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--full":
            run_full = True
        elif arg == "--n-trials" and i + 1 < len(args):
            n_trials = int(args[i + 1])

    print("Loading MNIST...")
    mnist = load_mnist()
    mnist = split_validation(mnist)

    # Use 5k subset for fast sweeping (~20-40s per trial)
    rng = np.random.default_rng(42)
    n_fast = 5000
    idx = rng.choice(mnist.train_images.shape[0], size=n_fast, replace=False)

    assert mnist.val_images is not None
    assert mnist.val_labels is not None

    data = {
        "train_images": mnist.train_images[idx],
        "train_labels": mnist.train_labels[idx],
        "val_images": mnist.val_images[:2000],
        "val_labels": mnist.val_labels[:2000],
    }

    print(f"Sweep: {n_fast} train, {data['val_labels'].shape[0]} val, {n_trials} trials\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="energy_sparse_coding_sweep")

    for i in range(n_trials):
        trial = study.ask()
        t0 = time.time()
        accuracy = objective(trial, data)
        elapsed = time.time() - t0

        study.tell(trial, accuracy)
        marker = " ***" if accuracy == study.best_value else ""
        print(
            f"  [{i+1:2d}/{n_trials}] {accuracy*100:5.1f}% ({elapsed:4.0f}s) "
            f"feat/cls={trial.params['n_features_per_class']:3d} "
            f"settle={trial.params['n_settle']:3d} "
            f"sparse={trial.params['sparsity']:.4f} "
            f"infer={trial.params['infer_rate']:.3f} "
            f"lr={trial.params['learn_rate']:.4f} "
            f"batch={trial.params['batch_size']:3d}"
            f"{marker}"
        )

    print(f"\n{'='*60}")
    print("  BEST HYPERPARAMETERS")
    print(f"{'='*60}")
    for k, v in study.best_params.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:>20s}: {val_str}")
    print(f"  {'val_acc':>20s}: {study.best_value:.4f} ({study.best_value*100:.1f}%)")

    if run_full:
        print(f"\n{'='*60}")
        print("  FULL EVALUATION WITH BEST PARAMS")
        print(f"{'='*60}")
        best = study.best_params
        model = SparseCodingV5Energy(
            n_features_per_class=best["n_features_per_class"],
            n_settle=best["n_settle"],
            sparsity=best["sparsity"],
            infer_rate=best["infer_rate"],
            learn_rate=best["learn_rate"],
            batch_size=best["batch_size"],
            epochs=30,
            seed=42,
        )
        model.train(mnist.train_images, mnist.train_labels)
        preds = model.predict(mnist.test_images)
        accuracy = float(np.mean(preds == mnist.test_labels))
        print(f"  Test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")


if __name__ == "__main__":
    main()
