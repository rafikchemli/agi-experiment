"""End-to-end runner for the causal dictionary learning POC.

Generates training events, trains a sparse dictionary, evaluates
specialization and compositionality, then produces visualizations and a
results summary.

Supports multiple architectures:
  - ista: Standard ISTA sparse coding (baseline)
  - product-of-experts: Factored rule/position codebooks
  - contrastive: ISTA + contrastive specialization pressure
  - contrastive-poe: PoE + contrastive on rule codebook

Usage:
    uv run python -m experiments.causal_dictionaries.run
    uv run python -m experiments.causal_dictionaries.run --arch contrastive
    uv run python -m experiments.causal_dictionaries.run --all-models
    uv run python -m experiments.causal_dictionaries.run --n-atoms 8 --sparsity 0.02
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
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
    ContrastiveProductOfExperts,
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

# Colours per architecture for comparison charts
_ARCH_COLORS = {
    "ista": "#E53935",
    "product-of-experts": "#FB8C00",
    "contrastive": "#1565C0",
    "contrastive-poe": "#6A1B9A",
}
_ARCH_LABELS = {
    "ista": "ISTA (baseline)",
    "product-of-experts": "ProductOfExperts",
    "contrastive": "Contrastive",
    "contrastive-poe": "ContrastivePoE",
}

RESULTS_DIR = Path(__file__).parent / "results"

_N_TESTS_PASS_RATIO = 0.75


def _pass_threshold(n_tests: int) -> int:
    """Return the minimum number of tests that must pass overall.

    Args:
        n_tests: Total number of composition tests.

    Returns:
        Minimum passing count (at least 4, or 75% of tests).
    """
    return max(4, int(n_tests * _N_TESTS_PASS_RATIO))


def _serialize_results(res: dict) -> dict:  # type: ignore[type-arg]
    """Serialise a results dict to JSON-compatible form.

    Args:
        res: Dict with keys ``history``, ``spec_scores``, ``test_results``.

    Returns:
        JSON-serialisable dict with summary statistics.
    """
    return {
        "tests_passed": sum(1 for t in res["test_results"] if t["passed"]),
        "mean_specialization": float(res["spec_scores"].mean()),
        "final_loss": res["history"][-1]["loss"],
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
            for t in res["test_results"]
        ],
    }


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
        encoding: Encoding scheme — ``"compact"``, ``"raw"``, or ``"learned"``.

    Returns:
        Tuple of (rule_data dict, shuffled training array,
        composition_data dict).
    """
    base_encoding = "raw" if encoding == "learned" else encoding
    click.echo(f"  Generating training events (encoding={encoding})...")
    all_rules = ["gravity", "containment", "contact", "bounce", "breakage"]
    rule_data: dict[str, np.ndarray] = {}
    for i, rule in enumerate(all_rules):
        events = generate_rule_events(rule, n_events=n_events, seed=seed + i)
        rule_data[rule] = encode_events_v2(events, encoding=base_encoding)
        click.echo(f"    {rule}: {len(events)} events")

    rng = np.random.default_rng(seed)
    all_data = np.vstack(list(rule_data.values()))
    shuffle_idx = rng.permutation(all_data.shape[0])
    all_data = all_data[shuffle_idx]

    click.echo("  Generating composition test events...")
    comp_data: dict[str, np.ndarray] = {}
    combos: list[tuple[str, list[str]]] = [
        ("T1 gravity+containment", ["gravity", "containment"]),
        ("T2 gravity+contact", ["gravity", "contact"]),
        ("T3 containment+contact", ["containment", "contact"]),
        ("T4 all original", ["gravity", "containment", "contact"]),
        ("T6 gravity+bounce", ["gravity", "bounce"]),
        ("T7 gravity+breakage", ["gravity", "breakage"]),
        ("T8 contact+bounce", ["gravity", "contact", "bounce"]),
        ("T9 contact+breakage", ["gravity", "contact", "breakage"]),
    ]
    for label, rules in combos:
        events = generate_composition_events(rules, n_events=200, seed=seed)
        comp_data[label] = encode_events_v2(events, encoding=base_encoding)
        click.echo(f"    {label}: {len(events)} events")

    negation_events = generate_rule_events("gravity", n_events=200, seed=seed + 100)
    comp_data["T5 negation"] = encode_events_v2(negation_events, encoding=base_encoding)
    click.echo(f"    T5 negation: {len(negation_events)} events")

    if encoding == "learned":
        click.echo("\n  Phase 1: Training autoencoder on raw features...")
        ae = LearnedEncoder(
            input_dim=all_data.shape[1],
            latent_dim=all_data.shape[1],
            hidden_dim=32,
            learn_rate=0.005,
            seed=seed,
        )
        ae.train(all_data, epochs=100, batch_size=64)
        for rule_name in rule_data:
            rule_data[rule_name] = ae.encode(rule_data[rule_name])
        all_data = ae.encode(all_data)
        for comp_name in comp_data:
            comp_data[comp_name] = ae.encode(comp_data[comp_name])
        click.echo(f"  Autoencoder output dim: {all_data.shape[1]}")

    return rule_data, all_data, comp_data


def _build_model(
    arch: str,
    n_atoms: int,
    sparsity: float,
    seed: int,
) -> DictionaryModel:
    """Build an untrained model for the given architecture.

    Args:
        arch: Architecture name (``"ista"``, ``"product-of-experts"``,
            ``"contrastive"``, ``"contrastive-poe"``).
        n_atoms: Number of dictionary atoms.
        sparsity: Sparsity penalty coefficient.
        seed: Random seed.

    Returns:
        An untrained model implementing the DictionaryModel protocol.

    Raises:
        click.ClickException: If the architecture name is not recognised.
    """
    if arch == "contrastive-poe":
        n_rule = n_atoms // 2
        n_pos = n_atoms - n_rule
        return ContrastiveProductOfExperts(
            n_rule_atoms=n_rule,
            n_pos_atoms=n_pos,
            sparsity=sparsity,
            contrastive_weight=1.0,
            seed=seed,
        )
    if arch == "product-of-experts":
        n_rule = n_atoms // 2
        n_pos = n_atoms - n_rule
        return ProductOfExperts(
            n_rule_atoms=n_rule,
            n_pos_atoms=n_pos,
            sparsity=sparsity,
            seed=seed,
        )
    if arch == "contrastive":
        return ContrastiveDictionary(
            n_atoms=n_atoms,
            sparsity=sparsity,
            contrastive_weight=2.0,
            seed=seed,
        )
    # ista
    return SparseDictionary(n_atoms=n_atoms, sparsity=sparsity, seed=seed)


def _train_model(
    arch: str,
    model: DictionaryModel,
    rule_data: dict[str, np.ndarray],
    all_data: np.ndarray,
    epochs: int,
) -> list[dict[str, float | int]]:
    """Train a model, dispatching to the right training method.

    Args:
        arch: Architecture name.
        model: Untrained model.
        rule_data: Per-rule encoded data (for contrastive models).
        all_data: Shuffled combined training data (for non-contrastive models).
        epochs: Number of training epochs.

    Returns:
        Training loss history.
    """
    if arch in ("contrastive", "contrastive-poe"):
        return model.train_with_labels(rule_data, epochs=epochs)  # type: ignore[union-attr]
    return model.train(all_data, epochs=epochs)  # type: ignore[union-attr]


def _run_composition_tests(
    sd: DictionaryModel,
    rule_data: dict[str, np.ndarray],
    comp_data: dict[str, np.ndarray],
) -> list[dict[str, str | float | bool]]:
    """Run composition tests T1–T9.

    Args:
        sd: Trained dictionary model.
        rule_data: Per-rule training data.
        comp_data: Composition test data.

    Returns:
        List of test result dicts with keys: name, ratio, jaccard, passed.
    """
    single_all = np.vstack(list(rule_data.values()))
    results: list[dict[str, str | float | bool]] = []

    rule_map_pair: dict[str, tuple[str, str]] = {
        "T1 gravity+containment": ("gravity", "containment"),
        "T2 gravity+contact": ("gravity", "contact"),
        "T3 containment+contact": ("containment", "contact"),
        "T6 gravity+bounce": ("gravity", "bounce"),
        "T7 gravity+breakage": ("gravity", "breakage"),
    }
    rule_map_multi: dict[str, tuple[list[str], list[str]]] = {
        "T4 all original": (["gravity"], ["containment", "contact"]),
        "T8 contact+bounce": (["contact", "gravity"], ["bounce"]),
        "T9 contact+breakage": (["contact", "gravity"], ["breakage"]),
    }

    for test_name, test_data in comp_data.items():
        ratio = composition_reconstruction_ratio(sd, single_all, test_data)

        if test_name == "T5 negation":
            passed = ratio < _RATIO_THRESHOLD
            results.append({
                "name": test_name,
                "ratio": ratio,
                "jaccard": float("nan"),
                "passed": passed,
            })
        elif test_name in rule_map_pair:
            rule_a_name, rule_b_name = rule_map_pair[test_name]
            jaccard = atom_union_jaccard(
                sd, rule_data[rule_a_name], rule_data[rule_b_name], test_data,
            )
            passed = ratio < _RATIO_THRESHOLD and jaccard >= _JACCARD_THRESHOLD
            results.append({
                "name": test_name,
                "ratio": ratio,
                "jaccard": jaccard,
                "passed": passed,
            })
        elif test_name in rule_map_multi:
            a_keys, b_keys = rule_map_multi[test_name]
            rule_a_data = np.vstack([rule_data[k] for k in a_keys])
            rule_b_data = np.vstack([rule_data[k] for k in b_keys])
            jaccard = atom_union_jaccard(sd, rule_a_data, rule_b_data, test_data)
            passed = ratio < _RATIO_THRESHOLD and jaccard >= _JACCARD_THRESHOLD
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
        sd: Trained dictionary model.
        rule_data: Per-rule training data.
        spec_scores: Per-atom specialization scores.
        test_results: Composition test results.
        n_events: Events per rule used for training.
        epochs: Training epochs.
    """
    affinity = atom_rule_affinity(sd, rule_data)
    rule_names = list(rule_data.keys())
    total_events = n_events * len(rule_names)

    best_rule_idx = np.argmax(affinity, axis=1)
    specialized_counts: dict[str, int] = {r: 0 for r in rule_names}
    shared_count = 0
    for i, score in enumerate(spec_scores):
        if score >= 0.6:
            specialized_counts[rule_names[best_rule_idx[i]]] += 1
        else:
            shared_count += 1

    n_tests = len(test_results)
    n_pass = sum(1 for t in test_results if t["passed"])
    overall = "PASS" if n_pass >= _pass_threshold(n_tests) else "FAIL"

    click.echo()
    click.echo("=" * 60)
    click.echo("  CAUSAL DICTIONARY POC RESULTS")
    click.echo("=" * 60)
    click.echo(
        f"  Dictionary: {sd.n_atoms} atoms, {epochs} epochs"
        f", {total_events} events"
    )
    click.echo()
    click.echo("  SPECIALIZATION")
    click.echo(f"  Mean specialization score:  {spec_scores.mean():.2f}")
    for rule_name in rule_names:
        label = f"Atoms specialized to {rule_name}:"
        click.echo(f"  {label:<35} {specialized_counts[rule_name]}")
    click.echo(f"  {'Atoms shared (no specialization):':<35} {shared_count}")
    click.echo()
    click.echo("  COMPOSITION TESTS")
    header = (
        f"  {'Test':<30} {'Recon Ratio':>11} "
        f"{'Jaccard':>9} {'Pass?':>6}"
    )
    click.echo(header)
    click.echo(
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
        click.echo(
            f"  {t['name']:<30} {t['ratio']:>11.2f} "
            f"{jaccard_str:>9} {pass_str:>6}"
        )
    click.echo()
    click.echo(f"  OVERALL: {overall} ({n_pass}/{n_tests} tests pass)")
    click.echo("=" * 60)


def _create_all_models_visualization(
    arch_results: dict[str, dict],  # type: ignore[type-arg]
    output_path: Path,
) -> None:
    """Create a comparison chart for all trained architectures.

    Produces a 3-panel figure:
    - Reconstruction ratios per test (grouped bars, lower is better)
    - Jaccard similarities per test (grouped bars, higher is better)
    - Pass/fail heatmap (architectures × tests)

    Args:
        arch_results: Mapping from arch name to results dict containing
            ``test_results`` and ``spec_scores``.
        output_path: Path to save the PNG figure.
    """
    archs = list(arch_results.keys())
    sample_tests = arch_results[archs[0]]["test_results"]
    test_names = [str(t["name"]) for t in sample_tests]
    n_tests = len(test_names)
    n_archs = len(archs)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        "All Architectures — Composition Test Comparison",
        fontsize=14, fontweight="bold", color=_CLR_LINES, y=1.01,
    )

    x_pos = np.arange(n_tests)
    bar_width = 0.8 / n_archs

    # ── Reconstruction ratios ──
    ax_ratio = axes[0]
    for i, arch in enumerate(archs):
        ratios = [float(t["ratio"]) for t in arch_results[arch]["test_results"]]
        offsets = x_pos + (i - n_archs / 2 + 0.5) * bar_width
        ax_ratio.bar(
            offsets, ratios, bar_width,
            label=_ARCH_LABELS.get(arch, arch),
            color=_ARCH_COLORS.get(arch, "#888"),
            alpha=0.8, edgecolor=_CLR_LINES, linewidth=0.5,
        )
    ax_ratio.axhline(
        y=_RATIO_THRESHOLD, color="#C62828", linestyle="--",
        linewidth=1.5, alpha=0.7, label=f"Fail threshold ({_RATIO_THRESHOLD})",
    )
    ax_ratio.set_xticks(x_pos)
    ax_ratio.set_xticklabels(
        [n.replace(" ", "\n") for n in test_names], fontsize=7,
    )
    ax_ratio.set_ylabel("Reconstruction Ratio", fontsize=10)
    ax_ratio.set_title("Reconstruction Ratio (lower is better)", fontsize=11, fontweight="bold")
    ax_ratio.legend(fontsize=8, loc="upper right")
    ax_ratio.grid(True, alpha=0.2, axis="y")

    # ── Jaccard similarities ──
    ax_jacc = axes[1]
    for i, arch in enumerate(archs):
        jaccards = [
            float(t["jaccard"]) if not np.isnan(float(t["jaccard"])) else 0.0
            for t in arch_results[arch]["test_results"]
        ]
        offsets = x_pos + (i - n_archs / 2 + 0.5) * bar_width
        ax_jacc.bar(
            offsets, jaccards, bar_width,
            label=_ARCH_LABELS.get(arch, arch),
            color=_ARCH_COLORS.get(arch, "#888"),
            alpha=0.8, edgecolor=_CLR_LINES, linewidth=0.5,
        )
    ax_jacc.axhline(
        y=_JACCARD_THRESHOLD, color="#2E7D32", linestyle="--",
        linewidth=1.5, alpha=0.7, label=f"Pass threshold ({_JACCARD_THRESHOLD})",
    )
    ax_jacc.set_xticks(x_pos)
    ax_jacc.set_xticklabels(
        [n.replace(" ", "\n") for n in test_names], fontsize=7,
    )
    ax_jacc.set_ylabel("Jaccard Similarity", fontsize=10)
    ax_jacc.set_title("Jaccard Similarity (higher is better)", fontsize=11, fontweight="bold")
    ax_jacc.legend(fontsize=8, loc="lower right")
    ax_jacc.grid(True, alpha=0.2, axis="y")
    ax_jacc.set_ylim(0, 1.15)

    # ── Pass/fail heatmap ──
    ax_heat = axes[2]
    heatmap = np.zeros((n_archs, n_tests))
    for i, arch in enumerate(archs):
        for j, t in enumerate(arch_results[arch]["test_results"]):
            heatmap[i, j] = 1.0 if t["passed"] else 0.0

    im = ax_heat.imshow(
        heatmap, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax_heat.set_xticks(range(n_tests))
    ax_heat.set_xticklabels(
        [n.replace(" ", "\n") for n in test_names], fontsize=7,
    )
    ax_heat.set_yticks(range(n_archs))
    arch_labels_short = [_ARCH_LABELS.get(a, a) for a in archs]
    ax_heat.set_yticklabels(arch_labels_short, fontsize=9)
    ax_heat.set_title("Pass / Fail Heatmap", fontsize=11, fontweight="bold")

    # Annotate cells with PASS/FAIL
    for i in range(n_archs):
        for j in range(n_tests):
            text = "PASS" if heatmap[i, j] == 1 else "FAIL"
            color = "black" if heatmap[i, j] == 1 else "white"
            ax_heat.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    # Summary: tests passed per arch
    pass_counts = [
        sum(1 for t in arch_results[a]["test_results"] if t["passed"])
        for a in archs
    ]
    for i, (arch, cnt) in enumerate(zip(archs, pass_counts, strict=True)):
        ax_heat.text(
            n_tests - 0.5, i, f"  {cnt}/{n_tests}",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color="#1565C0",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    click.echo(f"  All-models comparison saved: {output_path}")


def _create_comparison_visualization(
    ista_results: dict,  # type: ignore[type-arg]
    poe_results: dict,  # type: ignore[type-arg]
    rule_data: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Create a side-by-side comparison of ISTA baseline vs best architecture.

    Args:
        ista_results: Dict with keys: history, model, spec_scores, test_results.
        poe_results: Dict with keys: history, model, spec_scores, test_results.
        rule_data: Per-rule training data.
        output_path: Path to save the PNG figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Baseline (ISTA) vs Best Architecture — Raw Encoding",
        fontsize=14, fontweight="bold", color=_CLR_LINES, y=0.98,
    )

    ax_loss = axes[0, 0]
    for label, res, color in [
        ("ISTA (baseline)", ista_results, "#E53935"),
        ("Best arch", poe_results, "#1565C0"),
    ]:
        epochs_list = [h["epoch"] for h in res["history"]]
        losses = [h["loss"] for h in res["history"]]
        ax_loss.plot(epochs_list, losses, color=color, linewidth=2, label=label)
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("MSE Loss", fontsize=10)
    ax_loss.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

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
        label="Best arch", color="#1565C0", alpha=0.7,
        edgecolor=_CLR_LINES, linewidth=0.5,
    )
    ax_comp.axhline(
        y=_RATIO_THRESHOLD, color="#E53935", linestyle="--",
        linewidth=1.5, alpha=0.5, label=f"Fail threshold ({_RATIO_THRESHOLD})",
    )
    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels([n.replace(" ", "\n") for n in test_names], fontsize=7.5)
    ax_comp.set_ylabel("Reconstruction Ratio", fontsize=10)
    ax_comp.set_title(
        "Composition Tests — Reconstruction Ratio (lower is better)",
        fontsize=12, fontweight="bold",
    )
    ax_comp.legend(fontsize=8, loc="upper right")
    ax_comp.grid(True, alpha=0.2, axis="y")

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
        label="Best arch", color="#1565C0", alpha=0.7,
        edgecolor=_CLR_LINES, linewidth=0.5,
    )
    ax_jacc.axhline(
        y=_JACCARD_THRESHOLD, color="#43A047", linestyle="--",
        linewidth=1.5, alpha=0.5, label=f"Pass threshold ({_JACCARD_THRESHOLD})",
    )
    ax_jacc.set_xticks(x_pos)
    ax_jacc.set_xticklabels([n.replace(" ", "\n") for n in test_names], fontsize=7.5)
    ax_jacc.set_ylabel("Jaccard Similarity", fontsize=10)
    ax_jacc.set_title(
        "Composition Tests — Jaccard Similarity (higher is better)",
        fontsize=12, fontweight="bold",
    )
    ax_jacc.legend(fontsize=8, loc="lower right")
    ax_jacc.grid(True, alpha=0.2, axis="y")
    ax_jacc.set_ylim(0, 1.15)

    ax_tbl = axes[1, 1]
    ax_tbl.axis("off")

    ista_pass = sum(1 for t in ista_results["test_results"] if t["passed"])
    poe_pass = sum(1 for t in poe_results["test_results"] if t["passed"])
    n_tests = len(test_names)

    table_data = [
        ["", "ISTA (baseline)", "Best arch"],
        ["Tests passed", f"{ista_pass}/{n_tests}", f"{poe_pass}/{n_tests}"],
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

    for row_idx in range(1, len(table_data)):
        for col_idx in range(1, 3):
            cell = table[row_idx, col_idx]
            text = cell.get_text().get_text()
            if text == "PASS":
                cell.set_facecolor("#C8E6C9")
            elif text == "FAIL":
                cell.set_facecolor("#FFCDD2")
            elif "/" in text:
                passed, total = text.split("/")
                cell.set_facecolor(
                    "#C8E6C9" if int(passed) >= _pass_threshold(int(total)) else "#FFCDD2"
                )

    for col_idx in range(3):
        table[0, col_idx].set_facecolor("#E3F2FD")
        table[0, col_idx].set_text_props(fontweight="bold")

    ax_tbl.set_title("Results Summary", fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    click.echo(f"  Comparison visualization saved: {output_path}")


def _create_visualization(
    history: list[dict[str, float | int]],
    sd: DictionaryModel,
    rule_data: dict[str, np.ndarray],
    spec_scores: np.ndarray,
    test_results: list[dict[str, str | float | bool]],
    output_path: Path,
) -> None:
    """Create the 4-subplot results visualization figure.

    Args:
        history: Training loss history.
        sd: Trained dictionary model.
        rule_data: Per-rule training data.
        spec_scores: Per-atom specialization scores.
        test_results: Composition test results.
        output_path: Path to save the PNG figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Causal Dictionary Learning POC Results",
        fontsize=14, fontweight="bold", color=_CLR_LINES, y=0.98,
    )

    ax_loss = axes[0, 0]
    epoch_nums = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    ax_loss.plot(epoch_nums, losses, color=_CLR_LINES, linewidth=2, marker="o", markersize=3)
    ax_loss.fill_between(epoch_nums, losses, alpha=0.15, color="#1565C0")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("MSE Loss", fontsize=10)
    ax_loss.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax_loss.grid(True, alpha=0.3)

    ax_heat = axes[0, 1]
    affinity = atom_rule_affinity(sd, rule_data)
    rule_names = list(rule_data.keys())
    sort_idx = np.argsort(-spec_scores)
    affinity_sorted = affinity[sort_idx]

    im = ax_heat.imshow(affinity_sorted, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax_heat.set_xticks(range(len(rule_names)))
    ax_heat.set_xticklabels([r.capitalize() for r in rule_names], fontsize=9)
    ax_heat.set_ylabel("Atom (sorted by specialization)", fontsize=10)
    ax_heat.set_title("Atom-Rule Affinity", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax_heat, shrink=0.8, label="Mean activation")

    ax_comp = axes[1, 0]
    test_names = [str(t["name"]) for t in test_results]
    ratios = [float(t["ratio"]) for t in test_results]
    jaccards = [
        float(t["jaccard"]) if not np.isnan(float(t["jaccard"])) else 0.0
        for t in test_results
    ]

    x_pos = np.arange(len(test_names))
    bar_width = 0.35

    ax_comp.bar(
        x_pos - bar_width / 2, ratios, bar_width,
        label="Recon Ratio", color="#1565C0", alpha=0.7,
        edgecolor=_CLR_LINES, linewidth=0.5,
    )
    ax_comp.bar(
        x_pos + bar_width / 2, jaccards, bar_width,
        label="Jaccard", color="#2E7D32", alpha=0.7,
        edgecolor=_CLR_LINES, linewidth=0.5,
    )
    ax_comp.axhline(
        y=_RATIO_THRESHOLD, color="#E53935", linestyle="--",
        linewidth=1.5, alpha=0.7, label=f"Ratio threshold ({_RATIO_THRESHOLD})",
    )
    ax_comp.axhline(
        y=_JACCARD_THRESHOLD, color="#43A047", linestyle="--",
        linewidth=1.5, alpha=0.7, label=f"Jaccard threshold ({_JACCARD_THRESHOLD})",
    )
    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels([n.replace(" ", "\n") for n in test_names], fontsize=8, rotation=0)
    ax_comp.set_ylabel("Score", fontsize=10)
    ax_comp.set_title("Composition Tests", fontsize=12, fontweight="bold")
    ax_comp.legend(fontsize=8, loc="upper right")
    ax_comp.grid(True, alpha=0.2, axis="y")

    ax_hist = axes[1, 1]
    ax_hist.hist(spec_scores, bins=20, color="#1565C0", alpha=0.7, edgecolor=_CLR_LINES, linewidth=0.5)
    ax_hist.axvline(
        x=0.6, color="#E53935", linestyle="--",
        linewidth=2, alpha=0.8, label="Specialization threshold (0.6)",
    )
    ax_hist.set_xlabel("Specialization Score", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title("Atom Specialization Distribution", fontsize=12, fontweight="bold")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.2, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    click.echo(f"  Visualization saved: {output_path}")


@click.command()
@click.option(
    "--encoding",
    type=click.Choice(["compact", "raw", "learned"]),
    default="raw",
    show_default=True,
    help="Encoding scheme. 'raw' uses no derived features; 'learned' trains an autoencoder first.",
)
@click.option(
    "--arch",
    type=click.Choice(["ista", "product-of-experts", "contrastive", "contrastive-poe"]),
    default="contrastive",
    show_default=True,
    help="Architecture to use.",
)
@click.option("--n-atoms", type=int, default=10, show_default=True, help="Number of dictionary atoms.")
@click.option("--sparsity", type=float, default=0.05, show_default=True, help="Sparsity penalty.")
@click.option("--epochs", type=int, default=150, show_default=True, help="Number of training epochs.")
@click.option("--n-events", type=int, default=2000, show_default=True, help="Events per rule.")
@click.option("--compare", is_flag=True, help="Run ISTA baseline and chosen arch side-by-side.")
@click.option("--baseline", is_flag=True, help="Also train a baseline ISTA on mixed composition data.")
@click.option("--all-models", "all_models", is_flag=True,
              help="Train ISTA, PoE, and Contrastive and produce overall comparison chart.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
def main(  # noqa: PLR0912, PLR0913, PLR0915
    encoding: str,
    arch: str,
    n_atoms: int,
    sparsity: float,
    epochs: int,
    n_events: int,
    compare: bool,
    baseline: bool,
    all_models: bool,
    seed: int,
) -> None:
    """Run the full causal dictionary learning POC pipeline."""
    click.echo()
    click.echo("=" * 60)
    click.echo(f"  CAUSAL DICTIONARY LEARNING POC — {arch}")
    click.echo("=" * 60)
    click.echo()

    start_time = time.time()

    rule_data, all_data, comp_data = _generate_data(n_events, seed, encoding=encoding)

    click.echo(f"\n  Training {arch} ({n_atoms} atoms, sp={sparsity}, {epochs} epochs)...")
    model = _build_model(arch, n_atoms, sparsity, seed)
    history = _train_model(arch, model, rule_data, all_data, epochs)
    sd: DictionaryModel = model

    click.echo("\n  Computing specialization scores...")
    spec_scores = specialization_scores(sd, rule_data)

    click.echo("  Running composition tests...")
    test_results = _run_composition_tests(sd, rule_data, comp_data)

    _print_results(sd, rule_data, spec_scores, test_results, n_events, epochs)

    click.echo("\n  Generating visualization...")
    _create_visualization(history, sd, rule_data, spec_scores, test_results, RESULTS_DIR / "poc_results.png")

    n_tests = len(test_results)
    results_json: dict = {  # type: ignore[type-arg]
        "config": {
            "encoding": encoding,
            "architecture": arch,
            "n_atoms": n_atoms,
            "sparsity": sparsity,
            "epochs": epochs,
            "n_events_per_rule": n_events,
            "seed": seed,
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
                    float(t["jaccard"]) if not np.isnan(float(t["jaccard"])) else None
                ),
                "passed": bool(t["passed"]),
            }
            for t in test_results
        ],
        "overall_pass": sum(1 for t in test_results if t["passed"]) >= _pass_threshold(n_tests),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    results_path = RESULTS_DIR / "poc_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results_json, f, indent=2)
    click.echo(f"  Results JSON saved: {results_path}")

    # Optional: pairwise comparison (chosen arch vs ISTA baseline)
    if compare:
        click.echo("\n  === COMPARISON MODE ===")
        main_results: dict = {  # type: ignore[type-arg]
            "history": history,
            "model": sd,
            "spec_scores": spec_scores,
            "test_results": test_results,
        }

        if arch == "ista":
            click.echo("  Training Contrastive comparison...")
            cmp_sd = ContrastiveDictionary(n_atoms=n_atoms, sparsity=sparsity, contrastive_weight=2.0, seed=seed)
            cmp_history = cmp_sd.train_with_labels(rule_data, epochs=epochs)
            cmp_spec = specialization_scores(cmp_sd, rule_data)
            cmp_tests = _run_composition_tests(cmp_sd, rule_data, comp_data)
            ista_results = main_results
            arch_results_cmp = {
                "history": cmp_history, "model": cmp_sd,
                "spec_scores": cmp_spec, "test_results": cmp_tests,
            }
            arch_name = "contrastive"
        else:
            click.echo("  Training ISTA baseline...")
            ista_sd = SparseDictionary(n_atoms=n_atoms, sparsity=sparsity, seed=seed)
            ista_history = ista_sd.train(all_data, epochs=epochs)
            ista_spec = specialization_scores(ista_sd, rule_data)
            ista_tests = _run_composition_tests(ista_sd, rule_data, comp_data)
            ista_results = {
                "history": ista_history, "model": ista_sd,
                "spec_scores": ista_spec, "test_results": ista_tests,
            }
            arch_results_cmp = main_results
            arch_name = arch

        _create_comparison_visualization(ista_results, arch_results_cmp, rule_data, RESULTS_DIR / "comparison.png")

        results_json["comparison"] = {
            "ista": _serialize_results(ista_results),
            arch_name: _serialize_results(arch_results_cmp),
        }
        with results_path.open("w") as f:
            json.dump(results_json, f, indent=2)
        click.echo("  Comparison data added to results JSON.")

    # Optional: all-models comparison
    if all_models:
        click.echo("\n  === ALL MODELS COMPARISON ===")
        all_arch_results: dict[str, dict] = {}  # type: ignore[type-arg]

        # Use a reduced epoch count for speed in comparison mode
        cmp_epochs = min(epochs, 80)
        archs_to_compare = ["ista", "product-of-experts", "contrastive"]

        for cmp_arch in archs_to_compare:
            click.echo(f"\n  Training {cmp_arch}...")
            cmp_model = _build_model(cmp_arch, n_atoms, sparsity, seed)
            cmp_hist = _train_model(cmp_arch, cmp_model, rule_data, all_data, cmp_epochs)
            cmp_spec_sc = specialization_scores(cmp_model, rule_data)
            cmp_test_res = _run_composition_tests(cmp_model, rule_data, comp_data)
            all_arch_results[cmp_arch] = {
                "history": cmp_hist,
                "spec_scores": cmp_spec_sc,
                "test_results": cmp_test_res,
            }
            n_pass = sum(1 for t in cmp_test_res if t["passed"])
            click.echo(f"    {cmp_arch}: {n_pass}/{len(cmp_test_res)} tests passed")

        _create_all_models_visualization(all_arch_results, RESULTS_DIR / "all_models_comparison.png")

        results_json["all_models"] = {
            a: _serialize_results(r) for a, r in all_arch_results.items()
        }
        with results_path.open("w") as f:
            json.dump(results_json, f, indent=2)
        click.echo("  All-models data added to results JSON.")

    # Optional: baseline comparison (ISTA on composition data)
    if baseline:
        click.echo("\n  Training BASELINE ISTA on mixed composition...")
        comp_all = np.vstack(list(comp_data.values()))
        baseline_sd = SparseDictionary(n_atoms=n_atoms, sparsity=sparsity, seed=seed + 1000)
        baseline_history = baseline_sd.train(comp_all, epochs=epochs)
        baseline_spec = specialization_scores(baseline_sd, rule_data)
        click.echo(f"\n  Baseline final loss: {baseline_history[-1]['loss']:.6f}")
        click.echo(f"  Baseline mean specialization: {baseline_spec.mean():.2f}")
        click.echo(f"  Main dict mean specialization: {spec_scores.mean():.2f}")

    elapsed = time.time() - start_time
    click.echo(f"\n  Total elapsed: {elapsed:.1f}s")
    click.echo()


if __name__ == "__main__":
    main()
