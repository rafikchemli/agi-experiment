"""Interactive benchmark dashboard — Plotly-based HTML visualization.

Reads benchmark_results.json and generates an interactive HTML dashboard
with hover tooltips, toggleable traces, zoom, and pan.

Usage:
    uv run python -m benchmarks.interactive_dashboard
    # Opens benchmarks/results/dashboard.html in browser
"""

import json
import webbrowser
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_FILE = _RESULTS_DIR / "benchmark_results.json"
_DASHBOARD_FILE = _RESULTS_DIR / "dashboard.html"

# Approach colors — consistent with matplotlib evaluate.py palette
PALETTE: dict[str, str] = {
    "backprop_mlp": "#3B82F6",
    "forward_forward": "#F59E0B",
    "forward_forward_v2": "#D97706",
    "ff_enhanced": "#E11D48",
    "predictive_coding": "#10B981",
    "sparse_coding_v2": "#06B6D4",
    "sparse_coding_v4": "#059669",
    "sparse_coding_v5": "#7C3AED",
    "sparse_coding_v6": "#D946EF",
    "sparse_coding_v7": "#8B5CF6",
    "sparse_coding_v9": "#F43F5E",
    "sparse_coding_v13": "#EC4899",
    "sparse_coding_v15": "#14B8A6",
    "hybrid_v17": "#0EA5E9",
    "dfa_v20": "#22C55E",
}
DEFAULT_COLOR = "#64748B"

# Background
BG = "#FAFBFC"
GRID = "#E2E8F0"
TEXT = "#1E293B"


def _pretty(name: str) -> str:
    """Convert snake_case approach name to display name."""
    return name.replace("_", " ").title()


def _color(name: str) -> str:
    return PALETTE.get(name, DEFAULT_COLOR)


def build_dashboard(results: list[dict]) -> go.Figure:
    """Build the full interactive dashboard as a Plotly figure.

    Four panels:
    1. Final accuracy bar chart (top-left)
    2. Training accuracy curves (top-right)
    3. Loss curves (bottom-left)
    4. Accuracy vs training time scatter (bottom-right)

    Args:
        results: List of benchmark result dicts.

    Returns:
        Plotly Figure with all subplots.
    """
    sorted_results = sorted(results, key=lambda x: x["accuracy"])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Final Test Accuracy",
            "Training Accuracy Over Epochs",
            "Reconstruction / Loss Over Epochs",
            "Accuracy vs Training Time",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # ── Panel 1: Final accuracy bar chart ──
    names = [_pretty(r["name"]) for r in sorted_results]
    accs = [r["accuracy"] * 100 for r in sorted_results]
    times = [r["train_time"] for r in sorted_results]
    colors = [_color(r["name"]) for r in sorted_results]
    bp_tags = [
        "Backprop" if r["uses_backprop"] == "YES" else "Local learning"
        for r in sorted_results
    ]

    hover_texts = [
        f"<b>{n}</b><br>"
        f"Accuracy: {a:.2f}%<br>"
        f"Train time: {t:.0f}s<br>"
        f"Learning: {bp}"
        for n, a, t, bp in zip(names, accs, times, bp_tags)
    ]

    fig.add_trace(
        go.Bar(
            y=names,
            x=accs,
            orientation="h",
            marker_color=colors,
            text=[f"{a:.1f}%" for a in accs],
            textposition="outside",
            textfont_size=11,
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Chance line
    fig.add_vline(
        x=10, line_dash="dot", line_color="#EF4444", opacity=0.4, row=1, col=1
    )

    # ── Panel 2: Training accuracy curves ──
    mlp_acc = None
    for r in sorted_results:
        if r["name"] == "backprop_mlp":
            mlp_acc = r["accuracy"] * 100

    for r in sorted_results:
        hist = r.get("history", [])
        if not hist:
            continue

        epochs = [h["epoch"] for h in hist]
        train_accs = [h["train_acc"] * 100 for h in hist]
        color = _color(r["name"])
        display = _pretty(r["name"])

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_accs,
                mode="lines+markers",
                name=display,
                line={"color": color, "width": 2},
                marker={"size": 4},
                legendgroup=r["name"],
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "Epoch %{x}<br>"
                    "Train acc: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        # Validation accuracy if available
        val_accs = [h.get("val_acc") for h in hist]
        if any(v is not None for v in val_accs):
            val_epochs = [e for e, v in zip(epochs, val_accs) if v is not None]
            val_vals = [v * 100 for v in val_accs if v is not None]
            fig.add_trace(
                go.Scatter(
                    x=val_epochs,
                    y=val_vals,
                    mode="lines+markers",
                    name=f"{display} (val)",
                    line={"color": color, "width": 1.5, "dash": "dash"},
                    marker={"size": 3, "symbol": "square"},
                    legendgroup=r["name"],
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{display} (val)</b><br>"
                        "Epoch %{x}<br>"
                        "Val acc: %{y:.1f}%<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )

    # MLP baseline reference
    if mlp_acc is not None:
        fig.add_hline(
            y=mlp_acc,
            line_dash="dot",
            line_color=PALETTE["backprop_mlp"],
            opacity=0.5,
            annotation_text=f"MLP baseline {mlp_acc:.1f}%",
            annotation_font_size=10,
            annotation_font_color=PALETTE["backprop_mlp"],
            row=1,
            col=2,
        )

    # ── Panel 3: Loss curves ──
    for r in sorted_results:
        hist = r.get("history", [])
        if not hist:
            continue

        epochs = [h["epoch"] for h in hist]
        losses = [h.get("loss", 0) for h in hist]
        if all(lo == 0 for lo in losses):
            continue

        color = _color(r["name"])
        display = _pretty(r["name"])

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=losses,
                mode="lines+markers",
                name=display,
                line={"color": color, "width": 2},
                marker={"size": 3},
                legendgroup=r["name"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "Epoch %{x}<br>"
                    "Loss: %{y:.5f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    # ── Panel 4: Accuracy vs training time scatter ──
    for r in sorted_results:
        color = _color(r["name"])
        display = _pretty(r["name"])
        bp = "Backprop" if r["uses_backprop"] == "YES" else "Local"
        marker_symbol = "diamond" if r["uses_backprop"] == "YES" else "circle"

        fig.add_trace(
            go.Scatter(
                x=[r["train_time"]],
                y=[r["accuracy"] * 100],
                mode="markers+text",
                name=display,
                text=[display],
                textposition="top center",
                textfont_size=9,
                marker={
                    "color": color,
                    "size": 14,
                    "symbol": marker_symbol,
                    "line": {"width": 1.5, "color": "white"},
                },
                legendgroup=r["name"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    f"Accuracy: {r['accuracy'] * 100:.2f}%<br>"
                    f"Time: {r['train_time']:.0f}s<br>"
                    f"Learning: {bp}<br>"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

    # ── Layout ──
    fig.update_layout(
        title={
            "text": (
                "Brain-Inspired MNIST Benchmark Dashboard"
                "<br><sup>Non-backprop approaches for handwritten digit classification</sup>"
            ),
            "font": {"size": 20, "color": TEXT},
            "x": 0.5,
            "xanchor": "center",
        },
        height=850,
        width=1400,
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font={"color": TEXT, "family": "Inter, system-ui, sans-serif"},
        legend={
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": GRID,
            "borderwidth": 1,
            "font": {"size": 11},
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        hovermode="closest",
    )

    # Axis styling
    fig.update_xaxes(title_text="Accuracy (%)", gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(gridcolor=GRID, row=1, col=1)

    fig.update_xaxes(title_text="Epoch", gridcolor=GRID, row=1, col=2)
    fig.update_yaxes(
        title_text="Train Accuracy (%)",
        gridcolor=GRID,
        range=[0, 105],
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Epoch", gridcolor=GRID, row=2, col=1)
    fig.update_yaxes(
        title_text="Loss", gridcolor=GRID, type="log", row=2, col=1
    )

    fig.update_xaxes(
        title_text="Training Time (s)", gridcolor=GRID, row=2, col=2
    )
    fig.update_yaxes(
        title_text="Test Accuracy (%)", gridcolor=GRID, row=2, col=2
    )

    return fig


def main() -> None:
    """Load results and generate interactive dashboard."""
    if not _RESULTS_FILE.exists():
        print(f"No results found at {_RESULTS_FILE}")
        print("Run: uv run python -m benchmarks.evaluate")
        return

    with open(_RESULTS_FILE) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} approaches from {_RESULTS_FILE}")

    fig = build_dashboard(results)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(_DASHBOARD_FILE),
        include_plotlyjs=True,
        full_html=True,
        config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["toggleSpikelines"],
            "toImageButtonOptions": {
                "format": "png",
                "width": 1400,
                "height": 850,
                "scale": 2,
            },
        },
    )

    print(f"Dashboard saved to {_DASHBOARD_FILE}")
    webbrowser.open(f"file://{_DASHBOARD_FILE.resolve()}")


if __name__ == "__main__":
    main()
