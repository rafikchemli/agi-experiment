"""Representation quality tests — beyond raw accuracy.

Compares backprop vs non-backprop learning algorithms on properties
that accuracy alone doesn't capture:

1. Noise robustness: accuracy under increasing Gaussian noise
2. Graceful degradation: accuracy after knocking out random neurons/atoms
3. Occlusion robustness: accuracy with random pixel patches masked

The hypothesis (from PEPITA 2023, Bio-robustness 2025): bio-plausible
learning produces representations that are qualitatively different from
backprop — more robust, more distributed, more gracefully degrading.

Usage:
    uv run python -m benchmarks.representation_tests
"""

import copy
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.approaches.backprop_mlp import BackpropMLP
from benchmarks.approaches.dfa_v20 import DFAV20
from benchmarks.approaches.sparse_coding_v9_augmented import SparseCodingV9Augmented
from benchmarks.mnist_loader import load_mnist

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_noise_robustness(
    model: object,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    sigma_levels: list[float],
    seed: int = 42,
) -> dict[float, float]:
    """Measure accuracy under increasing Gaussian noise.

    Args:
        model: Trained model with predict() method.
        test_images: Clean test images, shape (N, 784).
        test_labels: True labels, shape (N,).
        sigma_levels: Noise standard deviations to test.
        seed: Random seed for reproducible noise.

    Returns:
        Dict mapping sigma → accuracy.
    """
    rng = np.random.default_rng(seed)
    results: dict[float, float] = {}

    for sigma in sigma_levels:
        if sigma == 0:
            noisy = test_images
        else:
            noise = rng.normal(0, sigma, test_images.shape)
            noisy = np.clip(test_images + noise, 0.0, 1.0)

        preds = model.predict(noisy)
        acc = float(np.mean(preds == test_labels))
        results[sigma] = acc

    return results


def test_degradation_mlp(
    model: object,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    dropout_rates: list[float],
    seed: int = 42,
) -> dict[float, float]:
    """Measure accuracy after zeroing random hidden neurons (layer 1).

    Works for both BackpropMLP (w1/b1) and DFAV20 (_w1/_b1).

    Args:
        model: Trained MLP/DFA model.
        test_images: Test images, shape (N, 784).
        test_labels: True labels, shape (N,).
        dropout_rates: Fraction of neurons to zero out.
        seed: Random seed.

    Returns:
        Dict mapping dropout_rate → accuracy.
    """
    rng = np.random.default_rng(seed)

    # Find weight attributes
    if hasattr(model, "_w1"):
        w_attr, b_attr = "_w1", "_b1"
    else:
        w_attr, b_attr = "w1", "b1"

    w_orig = getattr(model, w_attr).copy()
    b_orig = getattr(model, b_attr).copy()
    n_hidden = w_orig.shape[1]

    results: dict[float, float] = {}

    for rate in dropout_rates:
        if rate == 0:
            preds = model.predict(test_images)
        else:
            n_drop = int(n_hidden * rate)
            drop_idx = rng.choice(n_hidden, size=n_drop, replace=False)

            w_mod = w_orig.copy()
            b_mod = b_orig.copy()
            w_mod[:, drop_idx] = 0.0
            b_mod[drop_idx] = 0.0

            setattr(model, w_attr, w_mod)
            setattr(model, b_attr, b_mod)
            preds = model.predict(test_images)

            # Restore
            setattr(model, w_attr, w_orig.copy())
            setattr(model, b_attr, b_orig.copy())

        acc = float(np.mean(preds == test_labels))
        results[rate] = acc

    return results


def test_degradation_sparse(
    model: SparseCodingV9Augmented,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    dropout_rates: list[float],
    seed: int = 42,
) -> dict[float, float]:
    """Measure accuracy after zeroing random dictionary atoms.

    Args:
        model: Trained sparse coding model.
        test_images: Test images, shape (N, 784).
        test_labels: True labels, shape (N,).
        dropout_rates: Fraction of atoms to zero out per dictionary.
        seed: Random seed.

    Returns:
        Dict mapping dropout_rate → accuracy.
    """
    rng = np.random.default_rng(seed)
    orig_dicts = [d.copy() for d in model.dictionaries]
    n_atoms = orig_dicts[0].shape[1]

    results: dict[float, float] = {}

    for rate in dropout_rates:
        if rate == 0:
            preds = model.predict(test_images)
        else:
            n_drop = int(n_atoms * rate)
            for k in range(len(model.dictionaries)):
                model.dictionaries[k] = orig_dicts[k].copy()
                drop_idx = rng.choice(n_atoms, size=n_drop, replace=False)
                model.dictionaries[k][:, drop_idx] = 0.0

            preds = model.predict(test_images)

            # Restore
            for k in range(len(model.dictionaries)):
                model.dictionaries[k] = orig_dicts[k].copy()

        acc = float(np.mean(preds == test_labels))
        results[rate] = acc

    return results


def test_occlusion(
    model: object,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    occlusion_rates: list[float],
    patch_size: int = 4,
    seed: int = 42,
) -> dict[float, float]:
    """Measure accuracy with random square patches zeroed out.

    More biologically meaningful than Gaussian noise — simulates
    partial occlusion of the visual field.

    Args:
        model: Trained model with predict() method.
        test_images: Test images, shape (N, 784).
        test_labels: True labels, shape (N,).
        occlusion_rates: Fraction of pixels to occlude.
        patch_size: Size of square patches to zero out.
        seed: Random seed.

    Returns:
        Dict mapping occlusion_rate → accuracy.
    """
    rng = np.random.default_rng(seed)
    results: dict[float, float] = {}

    for rate in occlusion_rates:
        if rate == 0:
            preds = model.predict(test_images)
        else:
            occluded = test_images.copy().reshape(-1, 28, 28)
            n_patches = max(1, int(rate * 784 / (patch_size * patch_size)))

            for i in range(occluded.shape[0]):
                for _ in range(n_patches):
                    y = rng.integers(0, 28 - patch_size + 1)
                    x = rng.integers(0, 28 - patch_size + 1)
                    occluded[i, y : y + patch_size, x : x + patch_size] = 0.0

            preds = model.predict(occluded.reshape(-1, 784))

        acc = float(np.mean(preds == test_labels))
        results[rate] = acc

    return results


# ---------------------------------------------------------------------------
# Test 4: Adversarial robustness (transfer FGSM)
# ---------------------------------------------------------------------------


def _fgsm_gradient_mlp(
    model: BackpropMLP,
    images: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute gradient of cross-entropy loss w.r.t. input (for FGSM).

    Args:
        model: Trained BackpropMLP.
        images: Input images, shape (N, 784).
        labels: True labels, shape (N,).

    Returns:
        Gradient of loss w.r.t. input, shape (N, 784).
    """
    z1 = images @ model.w1 + model.b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ model.w2 + model.b2
    exp_z = np.exp(z2 - z2.max(axis=1, keepdims=True))
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)

    n = images.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0

    # Backprop to input
    dz2 = probs - one_hot
    da1 = dz2 @ model.w2.T
    dz1 = da1 * (z1 > 0).astype(np.float64)
    return dz1 @ model.w1.T


def test_adversarial(
    models: dict[str, object],
    backprop_model: BackpropMLP,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    epsilon_levels: list[float],
) -> dict[str, dict[float, float]]:
    """Transfer FGSM: generate adversarial examples from backprop, test all.

    Uses the backprop model's gradients to craft adversarial perturbations,
    then evaluates ALL models on these perturbations. This tests whether
    non-backprop models share backprop's adversarial vulnerabilities.

    Args:
        models: Dict of model_name → trained model.
        backprop_model: The backprop MLP used to generate FGSM gradients.
        test_images: Clean test images, shape (N, 784).
        test_labels: True labels, shape (N,).
        epsilon_levels: Perturbation magnitudes to test.

    Returns:
        Dict of model_name → {epsilon → accuracy}.
    """
    grad = _fgsm_gradient_mlp(backprop_model, test_images, test_labels)
    perturbation_dir = np.sign(grad)

    results: dict[str, dict[float, float]] = {}

    for name, model in models.items():
        results[name] = {}
        for eps in epsilon_levels:
            if eps == 0:
                adv_images = test_images
            else:
                adv_images = np.clip(
                    test_images + eps * perturbation_dir, 0.0, 1.0
                )
            preds = model.predict(adv_images)
            acc = float(np.mean(preds == test_labels))
            results[name][eps] = acc

    return results


# ---------------------------------------------------------------------------
# Test 5: Catastrophic forgetting (sequential class learning)
# ---------------------------------------------------------------------------


def _continue_training_mlp(
    model: BackpropMLP,
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 64,
) -> None:
    """Continue SGD training on new data without reinitializing weights.

    Args:
        model: Already-trained BackpropMLP.
        images: New training images.
        labels: New training labels.
        epochs: Number of additional epochs.
        lr: Learning rate for continued training.
        batch_size: Mini-batch size.
    """
    rng = np.random.default_rng(99)
    n = images.shape[0]
    targets = np.zeros((n, 10), dtype=np.float64)
    targets[np.arange(n), labels] = 1.0

    for _epoch in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x, y = images[idx], targets[idx]
            bs = len(idx)

            z1 = x @ model.w1 + model.b1
            a1 = np.maximum(0, z1)
            z2 = a1 @ model.w2 + model.b2
            exp_z = np.exp(z2 - z2.max(axis=1, keepdims=True))
            a2 = exp_z / exp_z.sum(axis=1, keepdims=True)

            dz2 = (a2 - y) / bs
            model.w2 -= lr * (a1.T @ dz2)
            model.b2 -= lr * np.sum(dz2, axis=0)
            da1 = dz2 @ model.w2.T
            dz1 = da1 * (z1 > 0).astype(np.float64)
            model.w1 -= lr * (x.T @ dz1)
            model.b1 -= lr * np.sum(dz1, axis=0)


def _continue_training_dfa(
    model: DFAV20,
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = 10,
    lr: float = 0.005,
    batch_size: int = 128,
) -> None:
    """Continue DFA training on new data without reinitializing.

    Args:
        model: Already-trained DFAV20.
        images: New training images.
        labels: New training labels.
        epochs: Number of additional epochs.
        lr: Learning rate for continued training.
        batch_size: Mini-batch size.
    """
    rng = np.random.default_rng(99)
    n = images.shape[0]
    one_hot = np.zeros((n, 10), dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0

    for _epoch in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x, y = images[idx], one_hot[idx]
            bs = x.shape[0]

            z1, h1, z2, h2, probs = model._forward(x)
            e = probs - y

            # Output layer (standard gradient)
            model._w_out -= lr * (h2.T @ e) / bs
            model._b_out -= lr * np.mean(e, axis=0)

            # Hidden 2 (DFA)
            delta2 = (e @ model._B2) * (z2 > 0).astype(np.float64)
            model._w2 -= lr * (h1.T @ delta2) / bs
            model._b2 -= lr * np.mean(delta2, axis=0)

            # Hidden 1 (DFA)
            delta1 = (e @ model._B1) * (z1 > 0).astype(np.float64)
            model._w1 -= lr * (x.T @ delta1) / bs
            model._b1 -= lr * np.mean(delta1, axis=0)


def _continue_training_sparse(
    model: SparseCodingV9Augmented,
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = 10,
) -> None:
    """Continue sparse coding dictionary learning on new data.

    Only dictionaries for classes present in labels get updated.
    Other dictionaries remain frozen — structural forgetting prevention.

    Args:
        model: Already-trained SparseCodingV9Augmented.
        images: New training images.
        labels: New training labels.
        epochs: Number of additional epochs.
    """
    rng = np.random.default_rng(99)
    n = images.shape[0]

    for _epoch in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, model.batch_size):
            idx = perm[start : start + model.batch_size]
            x_batch, y_batch = images[idx], labels[idx]

            for k in range(model.n_classes):
                mask = y_batch == k
                if mask.sum() < 2:  # noqa: PLR2004
                    continue

                x_k = x_batch[mask]
                bs = x_k.shape[0]
                z = model._settle(x_k, model.dictionaries[k])
                residual = x_k - z @ model.dictionaries[k].T
                model.dictionaries[k] += (
                    model.learn_rate * (residual.T @ z) / bs
                )
                model._apply_incoherence(k)
                norms = np.linalg.norm(
                    model.dictionaries[k], axis=0, keepdims=True
                )
                model.dictionaries[k] /= norms + 1e-8


def test_catastrophic_forgetting(
    models: dict[str, object],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    phase2_epochs: int = 10,
) -> dict[str, dict[str, float]]:
    """Test catastrophic forgetting via sequential class learning.

    Phase 1: Models are already trained on all 10 classes.
    Phase 2: Continue training on classes 5-9 ONLY for N epochs.
    Measure: how much accuracy on classes 0-4 drops (forgetting).

    Args:
        models: Dict of model_name → trained model (will be deep-copied).
        train_images: Full training set.
        train_labels: Full training labels.
        test_images: Full test set.
        test_labels: Full test labels.
        phase2_epochs: Epochs of phase 2 training on classes 5-9.

    Returns:
        Dict of model_name → {before_04, after_04, forgetting, after_59}.
    """
    # Split test data by class group
    mask_04 = test_labels <= 4
    mask_59 = test_labels >= 5

    # Phase 2 training data: classes 5-9 only
    train_mask_59 = train_labels >= 5
    train_59_images = train_images[train_mask_59]
    train_59_labels = train_labels[train_mask_59]

    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        # Measure accuracy BEFORE phase 2
        preds_before = model.predict(test_images)
        before_04 = float(np.mean(preds_before[mask_04] == test_labels[mask_04]))
        before_59 = float(np.mean(preds_before[mask_59] == test_labels[mask_59]))

        # Deep copy to avoid destroying the original model
        model_copy = copy.deepcopy(model)

        # Phase 2: continue training on classes 5-9 only
        if isinstance(model_copy, SparseCodingV9Augmented):
            _continue_training_sparse(
                model_copy, train_59_images, train_59_labels, phase2_epochs
            )
        elif isinstance(model_copy, DFAV20):
            _continue_training_dfa(
                model_copy, train_59_images, train_59_labels, phase2_epochs
            )
        else:
            _continue_training_mlp(
                model_copy, train_59_images, train_59_labels, phase2_epochs
            )

        # Measure accuracy AFTER phase 2
        preds_after = model_copy.predict(test_images)
        after_04 = float(np.mean(preds_after[mask_04] == test_labels[mask_04]))
        after_59 = float(np.mean(preds_after[mask_59] == test_labels[mask_59]))

        forgetting = before_04 - after_04

        results[name] = {
            "before_04": before_04,
            "after_04": after_04,
            "forgetting": forgetting,
            "before_59": before_59,
            "after_59": after_59,
        }

    return results


# ---------------------------------------------------------------------------
# Test 6: Confidence calibration
# ---------------------------------------------------------------------------


def _get_confidence_mlp(
    model: BackpropMLP, images: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and confidence (max softmax probability).

    Args:
        model: Trained BackpropMLP.
        images: Input images, shape (N, 784).

    Returns:
        Tuple of (predictions shape (N,), confidences shape (N,)).
    """
    z1 = images @ model.w1 + model.b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ model.w2 + model.b2
    exp_z = np.exp(z2 - z2.max(axis=1, keepdims=True))
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1).astype(np.uint8)
    confidence = np.max(probs, axis=1)
    return preds, confidence


def _get_confidence_dfa(
    model: DFAV20, images: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and confidence (max softmax probability).

    Args:
        model: Trained DFAV20.
        images: Input images, shape (N, 784).

    Returns:
        Tuple of (predictions shape (N,), confidences shape (N,)).
    """
    _, _, _, _, probs = model._forward(images)
    preds = np.argmax(probs, axis=1).astype(np.uint8)
    confidence = np.max(probs, axis=1)
    return preds, confidence


def _get_confidence_sparse(
    model: SparseCodingV9Augmented, images: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and confidence (reconstruction error margin).

    Confidence = (2nd_best_error - best_error) / 2nd_best_error.
    Ranges from 0 (tied) to 1 (best is much better than runner-up).

    Args:
        model: Trained SparseCodingV9Augmented.
        images: Input images, shape (N, 784).

    Returns:
        Tuple of (predictions shape (N,), confidences shape (N,)).
    """
    n = images.shape[0]
    errors = np.zeros((n, model.n_classes), dtype=np.float64)

    batch_sz = 1000
    for start in range(0, n, batch_sz):
        x_batch = images[start : start + batch_sz]
        bs = x_batch.shape[0]
        for k in range(model.n_classes):
            err = model._recon_error(x_batch, model.dictionaries[k])
            errors[start : start + bs, k] = err

    preds = np.argmin(errors, axis=1).astype(np.uint8)

    # Margin-based confidence: how much better is the best vs second-best
    sorted_errors = np.sort(errors, axis=1)
    best = sorted_errors[:, 0]
    second_best = sorted_errors[:, 1]
    confidence = (second_best - best) / (second_best + 1e-10)

    return preds, confidence


def test_calibration(
    models: dict[str, object],
    test_images: np.ndarray,
    test_labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, dict[str, float | list[float]]]:
    """Compute calibration metrics (ECE and per-bin accuracy).

    A well-calibrated model: when it says 80% confidence, it should be
    correct ~80% of the time. Backprop models are notoriously poorly
    calibrated (Guo et al. 2017).

    Args:
        models: Dict of model_name → trained model.
        test_images: Test images, shape (N, 784).
        test_labels: True labels, shape (N,).
        n_bins: Number of confidence bins.

    Returns:
        Dict of model_name → {ece, bin_confidences, bin_accuracies, bin_counts}.
    """
    results: dict[str, dict[str, float | list[float]]] = {}
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for name, model in models.items():
        # Get confidence scores
        if isinstance(model, SparseCodingV9Augmented):
            preds, conf = _get_confidence_sparse(model, test_images)
        elif isinstance(model, DFAV20):
            preds, conf = _get_confidence_dfa(model, test_images)
        else:
            preds, conf = _get_confidence_mlp(model, test_images)

        correct = (preds == test_labels).astype(np.float64)

        bin_accs: list[float] = []
        bin_confs: list[float] = []
        bin_counts: list[float] = []
        ece = 0.0
        n_total = len(preds)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (
                (conf >= lo) & (conf <= hi)
                if i == n_bins - 1
                else (conf >= lo) & (conf < hi)
            )

            n_bin = mask.sum()
            if n_bin == 0:
                bin_accs.append(0.0)
                bin_confs.append((lo + hi) / 2)
                bin_counts.append(0.0)
                continue

            avg_conf = float(np.mean(conf[mask]))
            avg_acc = float(np.mean(correct[mask]))
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
            bin_counts.append(float(n_bin))

            ece += (n_bin / n_total) * abs(avg_acc - avg_conf)

        results[name] = {
            "ece": ece,
            "bin_confidences": bin_confs,
            "bin_accuracies": bin_accs,
            "bin_counts": bin_counts,
        }

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_table(
    title: str,
    levels: list[float],
    level_label: str,
    results: dict[str, dict[float, float]],
) -> None:
    """Print a comparison table.

    Args:
        title: Table title.
        levels: The x-axis values (sigma, dropout rate, etc).
        level_label: Column header for the levels.
        results: Dict of model_name → {level → accuracy}.
    """
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")

    # Header
    header = f"{'Model':<22} | "
    header += " | ".join(f"{level_label}={v:<4}" for v in levels)
    header += " | Drop"
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        baseline = res[levels[0]]
        worst = res[levels[-1]]
        drop = baseline - worst

        row = f"{name:<22} | "
        row += " | ".join(f"{res[v]*100:>8.1f}%" for v in levels)
        row += f" | {drop*100:>+.1f}%"
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Train top 3 models and run representation quality battery."""
    print("Loading MNIST...")
    data = load_mnist()
    train_images = data.train_images
    train_labels = data.train_labels
    test_images = data.test_images
    test_labels = data.test_labels

    # Test parameters
    noise_sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    dropout_rates = [0.0, 0.1, 0.25, 0.5, 0.75]
    occlusion_rates = [0.0, 0.1, 0.2, 0.3, 0.5]

    # Models to test: backprop (benchmark), DFA, sparse coding
    models: dict[str, object] = {}
    clean_acc: dict[str, float] = {}

    # --- Train models ---

    print("\n" + "=" * 72)
    print("  TRAINING MODELS")
    print("=" * 72)

    # 1. Backprop MLP (benchmark)
    print("\n[1/3] Training BackpropMLP...")
    t0 = time.time()
    bp = BackpropMLP()
    bp.train(train_images, train_labels)
    bp_time = time.time() - t0
    preds = bp.predict(test_images)
    clean_acc["Backprop MLP"] = float(np.mean(preds == test_labels))
    models["Backprop MLP"] = bp
    print(f"  -> {clean_acc['Backprop MLP']*100:.1f}% ({bp_time:.0f}s)")

    # 2. DFA v20
    print("\n[2/3] Training DFA v20...")
    t0 = time.time()
    dfa = DFAV20()
    dfa.train(train_images, train_labels)
    dfa_time = time.time() - t0
    preds = dfa.predict(test_images)
    clean_acc["DFA v20"] = float(np.mean(preds == test_labels))
    models["DFA v20"] = dfa
    print(f"  -> {clean_acc['DFA v20']*100:.1f}% ({dfa_time:.0f}s)")

    # 3. Sparse Coding v9
    print("\n[3/3] Training Sparse Coding v9...")
    t0 = time.time()
    sc = SparseCodingV9Augmented()
    sc.train(train_images, train_labels)
    sc_time = time.time() - t0
    preds = sc.predict(test_images)
    clean_acc["Sparse Coding v9"] = float(np.mean(preds == test_labels))
    models["Sparse Coding v9"] = sc
    print(f"  -> {clean_acc['Sparse Coding v9']*100:.1f}% ({sc_time:.0f}s)")

    # --- Test battery ---

    print("\n" + "=" * 72)
    print("  RUNNING REPRESENTATION QUALITY TESTS")
    print("=" * 72)

    # Test 1: Noise robustness
    print("\nTest 1: Gaussian noise robustness...")
    noise_results: dict[str, dict[float, float]] = {}
    for name, model in models.items():
        print(f"  Testing {name}...")
        noise_results[name] = test_noise_robustness(
            model, test_images, test_labels, noise_sigmas
        )

    print_table("NOISE ROBUSTNESS (Gaussian)", noise_sigmas, "σ", noise_results)

    # Test 2: Graceful degradation
    print("Test 2: Graceful degradation (neuron/atom knockout)...")
    degrade_results: dict[str, dict[float, float]] = {}
    for name, model in models.items():
        print(f"  Testing {name}...")
        if isinstance(model, SparseCodingV9Augmented):
            degrade_results[name] = test_degradation_sparse(
                model, test_images, test_labels, dropout_rates
            )
        else:
            degrade_results[name] = test_degradation_mlp(
                model, test_images, test_labels, dropout_rates
            )

    print_table(
        "GRACEFUL DEGRADATION (neuron/atom knockout)",
        dropout_rates,
        "drop",
        degrade_results,
    )

    # Test 3: Occlusion robustness
    print("Test 3: Occlusion robustness (random 4x4 patches)...")
    occlusion_results: dict[str, dict[float, float]] = {}
    for name, model in models.items():
        print(f"  Testing {name}...")
        occlusion_results[name] = test_occlusion(
            model, test_images, test_labels, occlusion_rates
        )

    print_table(
        "OCCLUSION ROBUSTNESS (random 4x4 patches zeroed)",
        occlusion_rates,
        "occ",
        occlusion_results,
    )

    # Test 4: Adversarial robustness (transfer FGSM)
    print("Test 4: Adversarial robustness (transfer FGSM from backprop)...")
    adv_epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    adv_results = test_adversarial(
        models, bp, test_images, test_labels, adv_epsilons
    )
    print_table(
        "ADVERSARIAL ROBUSTNESS (transfer FGSM from backprop)",
        adv_epsilons,
        "ε",
        adv_results,
    )

    # Test 5: Confidence calibration
    print("Test 5: Confidence calibration (ECE)...")
    cal_results = test_calibration(models, test_images, test_labels)

    print(f"\n{'=' * 72}")
    print("  CONFIDENCE CALIBRATION")
    print(f"{'=' * 72}")
    print(f"{'Model':<22} | {'ECE':>8} | {'Interpretation'}")
    print("-" * 72)
    for name, cal in cal_results.items():
        ece = cal["ece"]
        interp = "well-calibrated" if ece < 0.05 else (  # noqa: PLR2004
            "moderate" if ece < 0.10 else "poorly calibrated"  # noqa: PLR2004
        )
        print(f"{name:<22} | {ece:>7.3f}  | {interp}")

    # Show calibration bins
    for name, cal in cal_results.items():
        print(f"\n  {name} calibration bins:")
        print(f"    {'Conf':>8} | {'Acc':>8} | {'Count':>8}")
        print(f"    {'-'*30}")
        bin_confs = cal["bin_confidences"]
        bin_accs = cal["bin_accuracies"]
        bin_counts = cal["bin_counts"]
        for i in range(len(bin_confs)):
            if bin_counts[i] > 0:
                print(
                    f"    {bin_confs[i]:>7.1%} | {bin_accs[i]:>7.1%} | "
                    f"{int(bin_counts[i]):>8d}"
                )

    # Test 6: Catastrophic forgetting
    print(f"\n{'=' * 72}")
    print("  Test 6: Catastrophic forgetting (10 epochs on classes 5-9 only)")
    print(f"{'=' * 72}")
    forget_results = test_catastrophic_forgetting(
        models, train_images, train_labels, test_images, test_labels,
        phase2_epochs=10,
    )

    print(f"\n{'=' * 72}")
    print("  CATASTROPHIC FORGETTING")
    print(f"{'=' * 72}")
    print(
        f"{'Model':<22} | {'0-4 Before':>10} | {'0-4 After':>10} | "
        f"{'Forgetting':>10} | {'5-9 After':>10}"
    )
    print("-" * 72)
    for name, res in forget_results.items():
        print(
            f"{name:<22} | {res['before_04']*100:>9.1f}% | "
            f"{res['after_04']*100:>9.1f}% | "
            f"{res['forgetting']*100:>+9.1f}% | "
            f"{res['after_59']*100:>9.1f}%"
        )

    # --- Grand summary ---
    print(f"\n{'=' * 72}")
    print("  GRAND SUMMARY")
    print(f"{'=' * 72}")
    print(
        f"{'Model':<22} | {'Clean':>7} | {'Noise':>7} | {'FGSM':>7} | "
        f"{'Degrade':>7} | {'Occlude':>7} | {'Forget':>7} | {'ECE':>7}"
    )
    print("-" * 82)

    for name in models:
        clean = clean_acc[name]
        noise_drop = clean - noise_results[name][noise_sigmas[-1]]
        adv_drop = clean - adv_results[name][adv_epsilons[-1]]
        degrade_drop = clean - degrade_results[name][dropout_rates[-1]]
        occlude_drop = clean - occlusion_results[name][occlusion_rates[-1]]
        forgetting = forget_results[name]["forgetting"]
        ece = cal_results[name]["ece"]

        print(
            f"{name:<22} | {clean*100:>6.1f}% | {noise_drop*100:>+6.1f}% | "
            f"{adv_drop*100:>+6.1f}% | {degrade_drop*100:>+6.1f}% | "
            f"{occlude_drop*100:>+6.1f}% | {forgetting*100:>+6.1f}% | "
            f"{ece:>6.3f}"
        )

    print()
    print("  Lower is better for all drop columns and ECE.")
    print("  Noise/FGSM/Degrade/Occlude = accuracy drop from clean → worst.")
    print("  Forget = accuracy drop on classes 0-4 after training on 5-9.")

    # Save results
    output = {
        "clean_accuracy": clean_acc,
        "noise_robustness": {
            k: {str(s): v for s, v in r.items()} for k, r in noise_results.items()
        },
        "degradation": {
            k: {str(s): v for s, v in r.items()} for k, r in degrade_results.items()
        },
        "occlusion": {
            k: {str(s): v for s, v in r.items()} for k, r in occlusion_results.items()
        },
        "adversarial": {
            k: {str(s): v for s, v in r.items()} for k, r in adv_results.items()
        },
        "calibration": {
            k: {"ece": v["ece"]} for k, v in cal_results.items()
        },
        "catastrophic_forgetting": forget_results,
    }

    out_path = Path("benchmarks/results/representation_tests.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
