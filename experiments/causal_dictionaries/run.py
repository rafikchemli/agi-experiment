"""End-to-end runner for the causal dictionary learning POC.

Generates training events, trains a sparse dictionary, evaluates
specialization and compositionality, then produces visualizations and a
results summary.

Supports multiple architectures:
  - ista: Standard ISTA sparse coding (baseline)
  - product-of-experts: Factored rule/position codebooks
  - contrastive: ISTA + contrastive specialization pressure

Usage:
    uv run python -m experiments.causal_dictionaries.run
    uv run python -m experiments.causal_dictionaries.run --arch contrastive
    uv run python -m experiments.causal_dictionaries.run --arch product-of-experts
    uv run python -m experiments.causal_dictionaries.run --n-atoms 8 --sparsity 0.02
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.causal_dictionaries.analysis import (
    DictionaryModel,
    atom_rule_affinity,
    atom_union_jaccard,
    composition_reconstruction_ratio,
    specialization_scores,
)
from experiments.causal_dictionaries.architectures import (
    ContrastiveDictionary,
    ProductOfExperts,
)
from experiments.causal_dictionaries.event_encoding import encode_events_v2
from experiments.causal_dictionaries.learned_encoder import LearnedEncoder
from experiments.causal_dictionaries.micro_world import (
    generate_composition_events,
    generate_rule_events,
)
from experiments.causal_dictionaries.sparse_dictionary import SparseDictionary

# Project color palette
_CLR_PRIMARY = "#E3F2FD"
_CLR_SUCCESS = "#E8F5E9"
_CLR_WARNING = "#FFF3E0"
_CLR_LINES = "#263238"
_CLR_SECONDARY = "#F3E5F5"

# Pass thresholds
_RATIO_THRESHOLD = 2.0
_JACCARD_THRESHOLD = 0.7

RESULTS_DIR = Path(__file__).parent / "results"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Causal dictionary learning POC runner.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="raw",
        choices=["compact", "raw", "learned"],
        help="Encoding scheme (default: raw). 'raw' uses no derived features; "
        "'learned' trains an autoencoder on raw features first.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="product-of-experts",
        choices=["ista", "product-of-experts", "contrastive"],
        help="Architecture to use (default: product-of-experts).",
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=6,
        help="Number of dictionary atoms (default: 6).",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.02,
        help="Sparsity penalty (default: 0.02).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs (default: 80).",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=2000,
        help="Number of events per rule (default: 2000).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both ISTA baseline and ProductOfExperts, produce comparison.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also train a baseline dictionary on mixed composition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def _generate_data(
    n_events: int,
    seed: int,
    encoding: str = "compact",
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
]:
    """Generate training and test data.

    Args:
        n_events: Number of events per rule.
        seed: Random seed.
        encoding: Encoding scheme — "compact", "raw", or "learned".

    Returns:
        Tuple of (rule_data dict, shuffled training array,
        composition_data dict).
    """
    # For learned encoding, first encode as raw, then transform
    base_encoding = "raw" if encoding == "learned" else encoding
    print(f"  Generating training events (encoding={encoding})...")
    rule_data: dict[str, np.ndarray] = {}
    for i, rule in enumerate(["gravity", "containment", "contact"]):
        events = generate_rule_events(
            rule, n_events=n_events, seed=seed + i
        )
        rule_data[rule] = encode_events_v2(events, encoding=base_encoding)
        print(f"    {rule}: {len(events)} events")

    # Shuffle all training data
    rng = np.random.default_rng(seed)
    all_data = np.vstack(list(rule_data.values()))
    shuffle_idx = rng.permutation(all_data.shape[0])
    all_data = all_data[shuffle_idx]

    # Generate composition test data
    print("  Generating composition test events...")
    comp_data: dict[str, np.ndarray] = {}
    combos: list[tuple[str, list[str]]] = [
        ("T1 gravity+containment", ["gravity", "containment"]),
        ("T2 gravity+contact", ["gravity", "contact"]),
        ("T3 containment+contact", ["containment", "contact"]),
        ("T4 all three", ["gravity", "containment", "contact"]),
    ]
    for label, rules in combos:
        events = generate_composition_events(
            rules, n_events=200, seed=seed
        )
        comp_data[label] = encode_events_v2(events, encoding=base_encoding)
        print(f"    {label}: {len(events)} events")

    # T5 negation: events from a rule NOT in the training mix
    # Use gravity events evaluated against containment atoms
    negation_events = generate_rule_events(
        "gravity", n_events=200, seed=seed + 100
    )
    comp_data["T5 negation"] = encode_events_v2(
        negation_events, encoding=base_encoding,
    )
    print(f"    T5 negation: {len(negation_events)} events")

    # Phase 1 (learned encoding only): train autoencoder on raw features
    if encoding == "learned":
        print("\n  Phase 1: Training autoencoder on raw features...")
        ae = LearnedEncoder(
            input_dim=all_data.shape[1],
            latent_dim=all_data.shape[1],
            hidden_dim=32,
            learn_rate=0.005,
            seed=seed,
        )
        ae.train(all_data, epochs=100, batch_size=64)
        # Transform all data through encoder bottleneck
        for rule_name in rule_data:
            rule_data[rule_name] = ae.encode(rule_data[rule_name])
        all_data = ae.encode(all_data)
        for comp_name in comp_data:
            comp_data[comp_name] = ae.encode(comp_data[comp_name])
        print(f"  Autoencoder output dim: {all_data.shape[1]}")

    return rule_data, all_data, comp_data


def _run_composition_tests(
    sd: DictionaryModel,
    rule_data: dict[str, np.ndarray],
    comp_data: dict[str, np.ndarray],
) -> list[dict[str, str | float | bool]]:
    """Run composition tests T1-T5.

    Args:
        sd: Trained SparseDictionary.
        rule_data: Per-rule training data.
        comp_data: Composition test data.

    Returns:
        List of test result dicts with keys: name, ratio, jaccard, passed.
    """
    single_all = np.vstack(list(rule_data.values()))
    results: list[dict[str, str | float | bool]] = []

    # Rule mapping for Jaccard computation
    rule_map: dict[str, tuple[str, str]] = {
        "T1 gravity+containment": ("gravity", "containment"),
        "T2 gravity+contact": ("gravity", "contact"),
        "T3 containment+contact": ("containment", "contact"),
    }

    for test_name, test_data in comp_data.items():
        ratio = composition_reconstruction_ratio(
            sd, single_all, test_data
        )

        if test_name == "T5 negation":
            # Negation test: only check ratio, Jaccard not applicable
            passed = ratio < _RATIO_THRESHOLD
            results.append({
                "name": test_name,
                "ratio": ratio,
                "jaccard": float("nan"),
                "passed": passed,
            })
        elif test_name == "T4 all three":
            # Use gravity + containment data as the two-rule baseline
            jaccard = atom_union_jaccard(
                sd,
                rule_data["gravity"],
                np.vstack([
                    rule_data["containment"],
                    rule_data["contact"],
                ]),
                test_data,
            )
            passed = ratio < _RATIO_THRESHOLD and jaccard > _JACCARD_THRESHOLD
            results.append({
                "name": test_name,
                "ratio": ratio,
                "jaccard": jaccard,
                "passed": passed,
            })
        else:
            rule_a_name, rule_b_name = rule_map[test_name]
            jaccard = atom_union_jaccard(
                sd,
                rule_data[rule_a_name],
                rule_data[rule_b_name],
                test_data,
            )
            passed = ratio < _RATIO_THRESHOLD and jaccard > _JACCARD_THRESHOLD
            results.append({
                "name": test_name,
                "ratio": ratio,
                "jaccard": jaccard,
                "passed": passed,
            })

    return results


def _print_results(
    sd: DictionaryModel,
    rule_data: dict[str, np.ndarray],
    spec_scores: np.ndarray,
    test_results: list[dict[str, str | float | bool]],
    n_events: int,
    epochs: int,
) -> None:
    """Print the results summary table to stdout.

    Args:
        sd: Trained SparseDictionary.
        rule_data: Per-rule training data.
        spec_scores: Per-atom specialization scores.
        test_results: Composition test results.
        n_events: Events per rule used for training.
        epochs: Training epochs.
    """
    affinity = atom_rule_affinity(sd, rule_data)
    rule_names = list(rule_data.keys())
    total_events = n_events * len(rule_names)

    # Count specialized atoms (threshold 0.6)
    best_rule_idx = np.argmax(affinity, axis=1)
    specialized_counts: dict[str, int] = {r: 0 for r in rule_names}
    shared_count = 0
    for i, score in enumerate(spec_scores):
        if score >= 0.6:
            specialized_counts[rule_names[best_rule_idx[i]]] += 1
        else:
            shared_count += 1

    n_pass = sum(1 for t in test_results if t["passed"])
    overall = "PASS" if n_pass >= 4 else "FAIL"

    print()
    print("=" * 60)
    print("  CAUSAL DICTIONARY POC RESULTS")
    print("=" * 60)
    print(
        f"  Dictionary: {sd.n_atoms} atoms, {epochs} epochs"
        f", {total_events} events"
    )
    print()
    print("  SPECIALIZATION")
    print(f"  Mean specialization score:  {spec_scores.mean():.2f}")
    for rule_name in rule_names:
        label = f"Atoms specialized to {rule_name}:"
        print(f"  {label:<35} {specialized_counts[rule_name]}")
    print(f"  {'Atoms shared (no specialization):':<35} {shared_count}")
    print()
    print("  COMPOSITION TESTS")
    header = (
        f"  {'Test':<30} {'Recon Ratio':>11} "
        f"{'Jaccard':>9} {'Pass?':>6}"
    )
    print(header)
    print(
        f"  {chr(0x2500) * 28} {chr(0x2500) * 11} "
        f"{chr(0x2500) * 9} {chr(0x2500) * 6}"
    )
    for t in test_results:
        jaccard_str = (
            f"{t['jaccard']:.2f}"
            if not (isinstance(t["jaccard"], float) and np.isnan(t["jaccard"]))
            else "  N/A"
        )
        pass_str = "YES" if t["passed"] else "NO"
        print(
            f"  {t['name']:<30} {t['ratio']:>11.2f} "
            f"{jaccard_str:>9} {pass_str:>6}"
        )
    print()
    print(f"  OVERALL: {overall} ({n_pass}/5 tests pass)")
    print("=" * 60)


def _create_comparison_visualization(
    ista_results: dict,
    poe_results: dict,
    rule_data: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Create a side-by-side comparison of ISTA baseline vs ProductOfExperts.

    Args:
        ista_results: Dict with keys: history, model, spec_scores, test_results.
        poe_results: Dict with keys: history, model, spec_scores, test_results.
        rule_data: Per-rule training data.
        output_path: Path to save the PNG figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Baseline (ISTA) vs ProductOfExperts — Raw Encoding",
        fontsize=14,
        fontweight="bold",
        color=_CLR_LINES,
        y=0.98,
    )

    # --- Top-left: Training loss curves (both) ---
    ax_loss = axes[0, 0]
    for label, res, color in [
        ("ISTA (baseline)", ista_results, "#E53935"),
        ("ProductOfExperts", poe_results, "#1565C0"),
    ]:
        epochs = [h["epoch"] for h in res["history"]]
        losses = [h["loss"] for h in res["history"]]
        ax_loss.plot(epochs, losses, color=color, linewidth=2, label=label)
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("MSE Loss", fontsize=10)
    ax_loss.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    # --- Top-right: Composition tests grouped bar chart ---
    ax_comp = axes[0, 1]
    test_names = [str(t["name"]) for t in ista_results["test_results"]]
    ista_ratios = [float(t["ratio"]) for t in ista_results["test_results"]]
    poe_ratios = [float(t["ratio"]) for t in poe_results["test_results"]]

    x_pos = np.arange(len(test_names))
    bar_width = 0.35

    ax_comp.bar(
        x_pos - bar_width / 2, ista_ratios, bar_width,
        label="ISTA (baseline)", color="#E53935", alpha=0.7,
        edgecolor=_CLR_LINES, linewidth=0.5,
    )
    ax_comp.bar(
        x_pos + bar_width / 2, poe_ratios, bar_width,
        label="ProductOfExperts", color="#1565C0", alpha=0.7,
        edgecolor=_CLR_LINES, linewidth=0.5,
    )
    ax_comp.axhline(
        y=_RATIO_THRESHOLD, color="#E53935", linestyle="--",
        linewidth=1.5, alpha=0.5, label=f"Fail threshold ({_RATIO_THRESHOLD})",
    )
    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels(
        [n.replace(" ", "\n") for n in test_names], fontsize=7.5,
    )
    ax_comp.set_ylabel("Reconstruction Ratio", fontsize=10)
    ax_comp.set_title(
        "Composition Tests — Reconstruction Ratio (lower is better)",
        fontsize=12, fontweight="bold",
    )
    ax_comp.legend(fontsize=8, loc="upper right")
    ax_comp.grid(True, alpha=0.2, axis="y")

    # --- Bottom-left: Jaccard comparison ---
    ax_jacc = axes[1, 0]
    ista_jaccards = [
        float(t["jaccard"]) if not np.isnan(float(t["jaccard"])) else 0.0
        for t in ista_results["test_results"]
    ]
    poe_jaccards = [
        float(t["jaccard"]) if not np.isnan(float(t["jaccard"])) else 0.0
        for t in poe_results["test_results"]
    ]
    ax_jacc.bar(
        x_pos - bar_width / 2, ista_jaccards, bar_width,
        label="ISTA (baseline)", color="#E53935", alpha=0.7,
        edgecolor=_CLR_LINES, linewidth=0.5,
    )
    ax_jacc.bar(
        x_pos + bar_width / 2, poe_jaccards, bar_width,
        label="ProductOfExperts", color="#1565C0", alpha=0.7,
        edgecolor=_CLR_LINES, linewidth=0.5,
    )
    ax_jacc.axhline(
        y=_JACCARD_THRESHOLD, color="#43A047", linestyle="--",
        linewidth=1.5, alpha=0.5, label=f"Pass threshold ({_JACCARD_THRESHOLD})",
    )
    ax_jacc.set_xticks(x_pos)
    ax_jacc.set_xticklabels(
        [n.replace(" ", "\n") for n in test_names], fontsize=7.5,
    )
    ax_jacc.set_ylabel("Jaccard Similarity", fontsize=10)
    ax_jacc.set_title(
        "Composition Tests — Jaccard Similarity (higher is better)",
        fontsize=12, fontweight="bold",
    )
    ax_jacc.legend(fontsize=8, loc="lower right")
    ax_jacc.grid(True, alpha=0.2, axis="y")
    ax_jacc.set_ylim(0, 1.15)

    # --- Bottom-right: Summary table ---
    ax_tbl = axes[1, 1]
    ax_tbl.axis("off")

    ista_pass = sum(1 for t in ista_results["test_results"] if t["passed"])
    poe_pass = sum(1 for t in poe_results["test_results"] if t["passed"])

    table_data = [
        ["", "ISTA (baseline)", "ProductOfExperts"],
        ["Tests passed", f"{ista_pass}/5", f"{poe_pass}/5"],
        [
            "Mean specialization",
            f"{ista_results['spec_scores'].mean():.2f}",
            f"{poe_results['spec_scores'].mean():.2f}",
        ],
        [
            "Final loss",
            f"{ista_results['history'][-1]['loss']:.4f}",
            f"{poe_results['history'][-1]['loss']:.4f}",
        ],
    ]
    for i, test_name in enumerate(test_names):
        ista_p = "PASS" if ista_results["test_results"][i]["passed"] else "FAIL"
        poe_p = "PASS" if poe_results["test_results"][i]["passed"] else "FAIL"
        table_data.append([test_name, ista_p, poe_p])

    table = ax_tbl.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Color pass/fail cells
    for row_idx in range(1, len(table_data)):
        for col_idx in range(1, 3):
            cell = table[row_idx, col_idx]
            text = cell.get_text().get_text()
            if text == "PASS":
                cell.set_facecolor("#C8E6C9")
            elif text == "FAIL":
                cell.set_facecolor("#FFCDD2")
            elif text.startswith("5/5"):
                cell.set_facecolor("#C8E6C9")
            elif "/" in text and not text.startswith("5/5"):
                cell.set_facecolor("#FFCDD2")

    # Header styling
    for col_idx in range(3):
        table[0, col_idx].set_facecolor("#E3F2FD")
        table[0, col_idx].set_text_props(fontweight="bold")

    ax_tbl.set_title(
        "Results Summary", fontsize=12, fontweight="bold", pad=20,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    print(f"  Comparison visualization saved: {output_path}")


def _create_visualization(
    history: list[dict[str, float | int]],
    sd: DictionaryModel,
    rule_data: dict[str, np.ndarray],
    spec_scores: np.ndarray,
    test_results: list[dict[str, str | float | bool]],
    output_path: Path,
) -> None:
    """Create the 4-subplot visualization figure.

    Args:
        history: Training loss history from SparseDictionary.train().
        sd: Trained SparseDictionary.
        rule_data: Per-rule training data.
        spec_scores: Per-atom specialization scores.
        test_results: Composition test results.
        output_path: Path to save the PNG figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Causal Dictionary Learning POC Results",
        fontsize=14,
        fontweight="bold",
        color=_CLR_LINES,
        y=0.98,
    )

    # --- Top-left: Training loss curve ---
    ax_loss = axes[0, 0]
    epoch_nums = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    ax_loss.plot(
        epoch_nums, losses,
        color=_CLR_LINES, linewidth=2, marker="o", markersize=3,
    )
    ax_loss.fill_between(
        epoch_nums, losses,
        alpha=0.15, color="#1565C0",
    )
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("MSE Loss", fontsize=10)
    ax_loss.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax_loss.grid(True, alpha=0.3)

    # --- Top-right: Atom-rule affinity heatmap ---
    ax_heat = axes[0, 1]
    affinity = atom_rule_affinity(sd, rule_data)
    rule_names = list(rule_data.keys())

    # Sort atoms by specialization (most specialized first)
    sort_idx = np.argsort(-spec_scores)
    affinity_sorted = affinity[sort_idx]

    im = ax_heat.imshow(
        affinity_sorted,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax_heat.set_xticks(range(len(rule_names)))
    ax_heat.set_xticklabels(
        [r.capitalize() for r in rule_names], fontsize=9
    )
    ax_heat.set_ylabel("Atom (sorted by specialization)", fontsize=10)
    ax_heat.set_title(
        "Atom-Rule Affinity", fontsize=12, fontweight="bold"
    )
    fig.colorbar(im, ax=ax_heat, shrink=0.8, label="Mean activation")

    # --- Bottom-left: Composition test results ---
    ax_comp = axes[1, 0]
    test_names = [str(t["name"]) for t in test_results]
    ratios = [float(t["ratio"]) for t in test_results]
    jaccards = [
        float(t["jaccard"]) if not np.isnan(float(t["jaccard"])) else 0.0
        for t in test_results
    ]

    x_pos = np.arange(len(test_names))
    bar_width = 0.35

    bars = ax_comp.bar(
        x_pos - bar_width / 2,
        ratios,
        bar_width,
        label="Recon Ratio",
        color="#1565C0",
        alpha=0.7,
        edgecolor=_CLR_LINES,
        linewidth=0.5,
    )
    ax_comp.bar(
        x_pos + bar_width / 2,
        jaccards,
        bar_width,
        label="Jaccard",
        color="#2E7D32",
        alpha=0.7,
        edgecolor=_CLR_LINES,
        linewidth=0.5,
    )

    # Threshold lines
    ax_comp.axhline(
        y=_RATIO_THRESHOLD, color="#E53935", linestyle="--",
        linewidth=1.5, alpha=0.7, label=f"Ratio threshold ({_RATIO_THRESHOLD})",
    )
    ax_comp.axhline(
        y=_JACCARD_THRESHOLD, color="#43A047", linestyle="--",
        linewidth=1.5, alpha=0.7, label=f"Jaccard threshold ({_JACCARD_THRESHOLD})",
    )

    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels(
        [n.replace(" ", "\n") for n in test_names],
        fontsize=8,
        rotation=0,
    )
    ax_comp.set_ylabel("Score", fontsize=10)
    ax_comp.set_title(
        "Composition Tests", fontsize=12, fontweight="bold"
    )
    ax_comp.legend(fontsize=8, loc="upper right")
    ax_comp.grid(True, alpha=0.2, axis="y")

    # --- Bottom-right: Specialization histogram ---
    ax_hist = axes[1, 1]
    ax_hist.hist(
        spec_scores,
        bins=20,
        color="#1565C0",
        alpha=0.7,
        edgecolor=_CLR_LINES,
        linewidth=0.5,
    )
    ax_hist.axvline(
        x=0.6, color="#E53935", linestyle="--",
        linewidth=2, alpha=0.8, label="Specialization threshold (0.6)",
    )
    ax_hist.set_xlabel("Specialization Score", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title(
        "Atom Specialization Distribution",
        fontsize=12,
        fontweight="bold",
    )
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.2, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print(f"  Visualization saved: {output_path}")


def main() -> None:
    """Run the full causal dictionary learning POC pipeline."""
    args = _parse_args()

    print()
    print("=" * 60)
    print(f"  CAUSAL DICTIONARY LEARNING POC — {args.arch}")
    print("=" * 60)
    print()

    start_time = time.time()

    # Step 1-2: Generate and encode events
    rule_data, all_data, comp_data = _generate_data(
        args.n_events, args.seed, encoding=args.encoding,
    )

    # Step 3: Build and train model
    arch_label = args.arch
    print(f"\n  Training {arch_label} ({args.n_atoms} atoms, "
          f"sp={args.sparsity}, {args.epochs} epochs)...")
    sd: DictionaryModel
    if args.arch == "product-of-experts":
        n_rule = args.n_atoms // 2
        n_pos = args.n_atoms - n_rule
        model = ProductOfExperts(
            n_rule_atoms=n_rule,
            n_pos_atoms=n_pos,
            sparsity=args.sparsity,
            seed=args.seed,
        )
        history = model.train(all_data, epochs=args.epochs)
        sd = model
    elif args.arch == "contrastive":
        model = ContrastiveDictionary(
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            contrastive_weight=0.5,
            seed=args.seed,
        )
        history = model.train_with_labels(rule_data, epochs=args.epochs)
        sd = model
    else:
        model = SparseDictionary(
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            seed=args.seed,
        )
        history = model.train(all_data, epochs=args.epochs)
        sd = model

    # Step 4: Compute specialization
    print("\n  Computing specialization scores...")
    spec_scores = specialization_scores(sd, rule_data)

    # Step 5-6: Run composition tests
    print("  Running composition tests...")
    test_results = _run_composition_tests(sd, rule_data, comp_data)

    # Step 7: Print results
    _print_results(
        sd, rule_data, spec_scores, test_results,
        args.n_events, args.epochs,
    )

    # Step 8: Generate visualization
    print("\n  Generating visualization...")
    _create_visualization(
        history, sd, rule_data, spec_scores, test_results,
        RESULTS_DIR / "poc_results.png",
    )

    # Step 9: Save results JSON
    results_json = {
        "config": {
            "encoding": args.encoding,
            "architecture": args.arch,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "epochs": args.epochs,
            "n_events_per_rule": args.n_events,
            "seed": args.seed,
        },
        "training": {
            "final_loss": history[-1]["loss"],
            "loss_history": [h["loss"] for h in history],
        },
        "specialization": {
            "mean_score": float(spec_scores.mean()),
            "scores": spec_scores.tolist(),
        },
        "composition_tests": [
            {
                "name": str(t["name"]),
                "ratio": float(t["ratio"]),
                "jaccard": (
                    float(t["jaccard"])
                    if not np.isnan(float(t["jaccard"]))
                    else None
                ),
                "passed": bool(t["passed"]),
            }
            for t in test_results
        ],
        "overall_pass": sum(
            1 for t in test_results if t["passed"]
        ) >= 4,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    results_path = RESULTS_DIR / "poc_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results JSON saved: {results_path}")

    # Optional: comparison mode (ISTA vs ProductOfExperts)
    if args.compare:
        print("\n  === COMPARISON MODE ===")
        # If we just ran ProductOfExperts, we already have its results
        # If we ran something else, train PoE now
        poe_results: dict = {}
        ista_results: dict = {}

        if args.arch == "product-of-experts":
            poe_results = {
                "history": history,
                "model": sd,
                "spec_scores": spec_scores,
                "test_results": test_results,
            }
            # Train ISTA baseline
            print("  Training ISTA baseline...")
            ista_sd = SparseDictionary(
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                seed=args.seed,
            )
            ista_history = ista_sd.train(all_data, epochs=args.epochs)
            ista_spec = specialization_scores(ista_sd, rule_data)
            ista_tests = _run_composition_tests(ista_sd, rule_data, comp_data)
            ista_results = {
                "history": ista_history,
                "model": ista_sd,
                "spec_scores": ista_spec,
                "test_results": ista_tests,
            }
        else:
            ista_results = {
                "history": history,
                "model": sd,
                "spec_scores": spec_scores,
                "test_results": test_results,
            }
            # Train ProductOfExperts
            print("  Training ProductOfExperts...")
            n_rule = args.n_atoms // 2
            n_pos = args.n_atoms - n_rule
            poe_sd = ProductOfExperts(
                n_rule_atoms=n_rule,
                n_pos_atoms=n_pos,
                sparsity=args.sparsity,
                seed=args.seed,
            )
            poe_history = poe_sd.train(all_data, epochs=args.epochs)
            poe_spec = specialization_scores(poe_sd, rule_data)
            poe_tests = _run_composition_tests(poe_sd, rule_data, comp_data)
            poe_results = {
                "history": poe_history,
                "model": poe_sd,
                "spec_scores": poe_spec,
                "test_results": poe_tests,
            }

        _create_comparison_visualization(
            ista_results, poe_results, rule_data,
            RESULTS_DIR / "comparison.png",
        )

        # Add comparison data to results JSON
        results_json["comparison"] = {
            "ista": {
                "tests_passed": sum(
                    1 for t in ista_results["test_results"] if t["passed"]
                ),
                "mean_specialization": float(
                    ista_results["spec_scores"].mean()
                ),
                "final_loss": ista_results["history"][-1]["loss"],
                "tests": [
                    {
                        "name": str(t["name"]),
                        "ratio": float(t["ratio"]),
                        "jaccard": (
                            float(t["jaccard"])
                            if not np.isnan(float(t["jaccard"]))
                            else None
                        ),
                        "passed": bool(t["passed"]),
                    }
                    for t in ista_results["test_results"]
                ],
            },
            "product_of_experts": {
                "tests_passed": sum(
                    1 for t in poe_results["test_results"] if t["passed"]
                ),
                "mean_specialization": float(
                    poe_results["spec_scores"].mean()
                ),
                "final_loss": poe_results["history"][-1]["loss"],
                "tests": [
                    {
                        "name": str(t["name"]),
                        "ratio": float(t["ratio"]),
                        "jaccard": (
                            float(t["jaccard"])
                            if not np.isnan(float(t["jaccard"]))
                            else None
                        ),
                        "passed": bool(t["passed"]),
                    }
                    for t in poe_results["test_results"]
                ],
            },
        }
        # Re-save with comparison data
        with results_path.open("w") as f:
            json.dump(results_json, f, indent=2)
        print("  Comparison data added to results JSON.")

    # Optional: baseline comparison (ISTA on composition data)
    if args.baseline:
        print("\n  Training BASELINE ISTA on mixed composition...")
        comp_all = np.vstack(list(comp_data.values()))
        baseline_sd = SparseDictionary(
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            seed=args.seed + 1000,
        )
        baseline_history = baseline_sd.train(
            comp_all, epochs=args.epochs
        )
        baseline_spec = specialization_scores(baseline_sd, rule_data)
        print(
            f"\n  Baseline final loss: "
            f"{baseline_history[-1]['loss']:.6f}"
        )
        print(
            f"  Baseline mean specialization: "
            f"{baseline_spec.mean():.2f}"
        )
        print(
            f"  Main dict mean specialization: "
            f"{spec_scores.mean():.2f}"
        )

    elapsed = time.time() - start_time
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
