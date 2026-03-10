"""Evaluation runner — train approaches on MNIST and compare results.

Usage:
    uv run python -m benchmarks.evaluate                    # Run all approaches
    uv run python -m benchmarks.evaluate backprop_mlp       # Run one approach
    uv run python -m benchmarks.evaluate --plot             # Plot saved results
"""

import importlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.base import MNISTApproach
from benchmarks.mnist_loader import load_mnist, split_validation

_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_FILE = _RESULTS_DIR / "benchmark_results.json"


def evaluate_approach(
    approach: MNISTApproach,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float | str]:
    """Train and evaluate a single approach.

    Args:
        approach: The approach to evaluate.
        train_images: Training images (N, 784).
        train_labels: Training labels (N,).
        test_images: Test images (M, 784).
        test_labels: Test labels (M,).

    Returns:
        Dict with name, accuracy, train_time, uses_backprop.
    """
    print(f"\n{'='*60}")
    print(f"  {approach.name}")
    print(f"  Backprop: {'YES' if approach.uses_backprop else 'NO'}")
    print(f"{'='*60}")

    # Train
    t0 = time.time()
    approach.train(train_images, train_labels)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Predict
    predictions = approach.predict(test_images)
    accuracy = float(np.mean(predictions == test_labels))
    print(f"  Test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    return {
        "name": approach.name,
        "accuracy": accuracy,
        "train_time": train_time,
        "uses_backprop": "YES" if approach.uses_backprop else "NO",
        "history": [m.to_dict() for m in approach.history],
    }


def print_comparison(results: list[dict[str, float | str]]) -> None:
    """Print a comparison table of all evaluated approaches.

    Args:
        results: List of result dicts from evaluate_approach.
    """
    print(f"\n{'='*60}")
    print("  MNIST Benchmark Results")
    print(f"{'='*60}")
    print(f"  {'Approach':<25} {'Accuracy':>10} {'Time':>10} {'Backprop?':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for r in sorted(results, key=lambda x: -x["accuracy"]):  # type: ignore[arg-type]
        print(
            f"  {r['name']:<25} {r['accuracy']:>9.1%} {r['train_time']:>9.1f}s {r['uses_backprop']:>10}"
        )
    print()


def get_available_approaches() -> dict[str, type[MNISTApproach]]:
    """Auto-discover all MNISTApproach subclasses in benchmarks/approaches/.

    Scans every .py file in the approaches directory, imports it, and
    collects any class that inherits from MNISTApproach. This means
    adding a new approach is as simple as dropping a new file — no
    registration needed.

    Returns:
        Dict mapping approach name to class.
    """
    approaches: dict[str, type[MNISTApproach]] = {}
    approaches_dir = Path(__file__).parent / "approaches"

    for py_file in sorted(approaches_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"benchmarks.approaches.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, MNISTApproach)
                and obj is not MNISTApproach
                and hasattr(obj, "name")
            ):
                approaches[obj.name] = obj

    return approaches


def save_results(new_results: list[dict[str, float | str]]) -> Path:
    """Save benchmark results, merging with any previously saved results.

    Results are keyed by name — re-running an approach updates its entry
    rather than duplicating it.

    Args:
        new_results: List of result dicts from the current run.

    Returns:
        Path to the saved JSON file.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results
    existing: dict[str, dict[str, float | str]] = {}
    if _RESULTS_FILE.exists():
        with open(_RESULTS_FILE) as f:
            for r in json.load(f):
                existing[r["name"]] = r

    # Upsert new results
    for r in new_results:
        existing[r["name"]] = r

    merged = list(existing.values())
    with open(_RESULTS_FILE, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"  Results saved to {_RESULTS_FILE} ({len(merged)} approach(es))")
    return _RESULTS_FILE


def load_results() -> list[dict[str, float | str]]:
    """Load previously saved benchmark results.

    Returns:
        List of result dicts.
    """
    with open(_RESULTS_FILE) as f:
        return json.load(f)


def plot_results(results: list[dict[str, float | str]]) -> None:
    """Generate benchmark plots: accuracy bars + training curves.

    Two-panel figure:
    - Left: final accuracy bar chart (all approaches)
    - Right: training curves (accuracy vs epoch) for approaches with history

    Args:
        results: List of result dicts from evaluate_approach.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # --- Style ---
    c_bg = "#F8FAFC"
    c_grid = "#E2E8F0"
    c_text = "#1E293B"
    # Per-approach colors
    palette = {
        "backprop_mlp": "#3B82F6",
        "forward_forward": "#F59E0B",
        "ff_enhanced": "#E11D48",
        "predictive_coding": "#10B981",
        "sparse_coding_v4": "#059669",
        "sparse_coding_v2": "#06B6D4",
        "sparse_coding_v5": "#7C3AED",
        "sparse_coding_v6": "#D946EF",
        "sparse_coding_v9": "#F43F5E",
        "stdp_snn": "#EF4444",
        "hebbian": "#8B5CF6",
    }
    default_color = "#64748B"

    results_sorted = sorted(results, key=lambda x: x["accuracy"])  # type: ignore[arg-type]
    has_history = any(r.get("history") for r in results_sorted)

    # Determine layout
    if has_history:
        fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    else:
        fig, ax_bar = plt.subplots(figsize=(8, max(3.5, 1 + len(results_sorted) * 0.7)))
        ax_curve = None

    fig.patch.set_facecolor(c_bg)

    # ---- LEFT: Accuracy bar chart ----
    ax_bar.set_facecolor(c_bg)
    names = [r["name"].replace("_", " ").title() for r in results_sorted]
    accuracies = [float(r["accuracy"]) * 100 for r in results_sorted]
    times = [float(r["train_time"]) for r in results_sorted]
    colors = [palette.get(r["name"], default_color) for r in results_sorted]

    y_pos = np.arange(len(results_sorted))
    bars = ax_bar.barh(y_pos, accuracies, height=0.55, color=colors, edgecolor="white", linewidth=1.5)

    # Chance reference
    ax_bar.axvline(x=10, color="#EF4444", linestyle=":", linewidth=1, alpha=0.4)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=10, color=c_text)
    ax_bar.set_xlabel("Test Accuracy (%)", fontsize=10, color=c_text)
    ax_bar.set_xlim(0, 110)
    ax_bar.set_title("Final Accuracy", fontsize=12, fontweight="bold", color=c_text, pad=12)

    for bar, acc, t, bp in zip(bars, accuracies, times, [r["uses_backprop"] for r in results_sorted]):
        bp_tag = "" if bp == "YES" else " (local)"
        ax_bar.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%  {t:.0f}s{bp_tag}",
            va="center", fontsize=9, color=c_text,
        )

    # Clean spines
    for spine in ("top", "right", "left"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["bottom"].set_color(c_grid)
    ax_bar.tick_params(left=False, colors=c_text)
    ax_bar.xaxis.grid(True, color=c_grid, linewidth=0.5, alpha=0.7)
    ax_bar.set_axisbelow(True)

    # ---- RIGHT: Training curves ----
    if ax_curve is not None:
        ax_curve.set_facecolor(c_bg)

        # Find MLP final accuracy for reference line
        mlp_acc = None
        for r in results_sorted:
            if r["name"] == "backprop_mlp":
                mlp_acc = float(r["accuracy"]) * 100

        for r in results_sorted:
            hist = r.get("history", [])
            if not hist:
                continue
            epochs = [h["epoch"] for h in hist]
            train_accs = [h["train_acc"] * 100 for h in hist]
            color = palette.get(r["name"], default_color)
            label = r["name"].replace("_", " ").title()
            ax_curve.plot(epochs, train_accs, color=color, linewidth=2, label=label, marker="o", markersize=3)

            # Val accuracy if available
            val_accs = [h.get("val_acc") for h in hist]
            if any(v is not None for v in val_accs):
                val_epochs = [e for e, v in zip(epochs, val_accs) if v is not None]
                val_vals = [v * 100 for v in val_accs if v is not None]
                ax_curve.plot(
                    val_epochs, val_vals, color=color, linewidth=1.5,
                    linestyle="--", alpha=0.7, marker="s", markersize=2,
                )

        # MLP baseline reference line
        if mlp_acc is not None:
            ax_curve.axhline(y=mlp_acc, color=palette["backprop_mlp"], linestyle=":", linewidth=1.5, alpha=0.5)
            ax_curve.text(
                1, mlp_acc + 0.5,
                f"MLP baseline {mlp_acc:.1f}%",
                color=palette["backprop_mlp"], fontsize=8, alpha=0.7,
            )

        # Chance line
        ax_curve.axhline(y=10, color="#EF4444", linestyle=":", linewidth=1, alpha=0.3)
        ax_curve.text(1, 11.5, "chance", color="#EF4444", fontsize=8, alpha=0.5)

        ax_curve.set_xlabel("Epoch", fontsize=10, color=c_text)
        ax_curve.set_ylabel("Accuracy (%)", fontsize=10, color=c_text)
        ax_curve.set_title("Training Progress", fontsize=12, fontweight="bold", color=c_text, pad=12)
        ax_curve.set_ylim(0, 105)
        ax_curve.legend(fontsize=9, frameon=True, facecolor="white", edgecolor=c_grid, framealpha=0.9)

        for spine in ("top", "right"):
            ax_curve.spines[spine].set_visible(False)
        ax_curve.spines["left"].set_color(c_grid)
        ax_curve.spines["bottom"].set_color(c_grid)
        ax_curve.tick_params(colors=c_text)
        ax_curve.yaxis.grid(True, color=c_grid, linewidth=0.5, alpha=0.7)
        ax_curve.set_axisbelow(True)

    plt.tight_layout()

    plot_path = _RESULTS_DIR / "benchmark_comparison.png"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=c_bg)
    print(f"  Plot saved to {plot_path}")
    plt.close(fig)


def main() -> None:
    """Run benchmark evaluation."""
    print("Loading MNIST...")
    data = load_mnist()
    data = split_validation(data)
    print(
        f"  Train: {data.train_images.shape[0]} images, "
        f"Val: {data.val_images.shape[0] if data.val_images is not None else 0} images, "
        f"Test: {data.test_images.shape[0]} images"
    )

    available = get_available_approaches()
    if not available:
        print("No approaches found! Add classes to benchmarks/approaches/")
        sys.exit(1)

    # Filter by command-line argument if provided
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        name = args[0]
        if name not in available:
            print(f"Unknown approach: {name}")
            print(f"Available: {', '.join(available.keys())}")
            sys.exit(1)
        available = {name: available[name]}

    print(f"Running {len(available)} approach(es): {', '.join(available.keys())}")

    results = []
    for name, cls in available.items():
        approach = cls()

        # Wire up validation data for approaches that support it
        if hasattr(approach, "set_validation") and data.val_images is not None:
            approach.set_validation(data.val_images, data.val_labels)

        result = evaluate_approach(
            approach,
            data.train_images,
            data.train_labels,
            data.test_images,
            data.test_labels,
        )
        results.append(result)

    if len(results) > 1:
        print_comparison(results)

    save_results(results)

    # Plot all accumulated results
    all_results = load_results()
    if len(all_results) > 1:
        print_comparison(all_results)
    plot_results(all_results)


if __name__ == "__main__":
    # Quick --plot flag to re-plot saved results without re-training
    if "--plot" in sys.argv:
        results = load_results()
        print_comparison(results)
        plot_results(results)
    else:
        main()
