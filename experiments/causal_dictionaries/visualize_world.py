"""Visualize the micro-world simulator — grid states and event transitions.

Usage:
    uv run python -m experiments.causal_dictionaries.visualize_world
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from experiments.causal_dictionaries.micro_world import Event, GridWorld, generate_rule_events

# Object display config
OBJ_COLORS = {
    "ball": "#E53935",
    "cup": "#1E88E5",
    "box": "#43A047",
    "shelf": "#8D6E63",
    "table": "#6D4C41",
}
OBJ_MARKERS = {
    "ball": "o",
    "cup": "v",
    "box": "s",
    "shelf": "_",
    "table": "_",
}

GRID_SIZE = 5


def _draw_grid(ax: plt.Axes, title: str = "") -> None:
    """Draw the 5x5 grid background."""
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(range(GRID_SIZE), fontsize=8)
    ax.set_yticklabels(range(GRID_SIZE), fontsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xlabel("col", fontsize=8)
    ax.set_ylabel("row (0=floor)", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    # Floor line
    ax.axhline(y=-0.3, color="#8D6E63", linewidth=3, alpha=0.5)
    ax.text(
        GRID_SIZE / 2 - 0.5, -0.45, "floor",
        fontsize=7, color="#8D6E63", alpha=0.7, ha="center",
    )


def _draw_object(
    ax: plt.Axes,
    name: str,
    obj_type: str,
    row: int,
    col: int,
    alpha: float = 1.0,
    label: bool = True,
) -> None:
    """Draw an object on the grid."""
    color = OBJ_COLORS.get(obj_type, "#999999")

    if obj_type in ("table", "shelf"):
        # Draw as a horizontal platform
        rect = patches.FancyBboxPatch(
            (col - 0.35, row - 0.1), 0.7, 0.2,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="black",
            alpha=alpha, linewidth=1.5,
        )
        ax.add_patch(rect)
        if label:
            ax.text(
                col, row + 0.25, name,
                fontsize=7, ha="center", va="bottom", color=color, alpha=alpha,
            )
    elif obj_type == "box":
        rect = patches.FancyBboxPatch(
            (col - 0.25, row - 0.25), 0.5, 0.5,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="black",
            alpha=alpha, linewidth=1.5,
        )
        ax.add_patch(rect)
        if label:
            ax.text(
                col, row + 0.35, name,
                fontsize=7, ha="center", va="bottom", color=color, alpha=alpha,
            )
    else:
        # ball, cup — draw as marker
        marker = OBJ_MARKERS.get(obj_type, "o")
        ax.plot(
            col, row, marker=marker, markersize=18,
            color=color, markeredgecolor="black",
            markeredgewidth=1.5, alpha=alpha,
        )
        if label:
            ax.text(
                col, row + 0.35, name,
                fontsize=7, ha="center", va="bottom", color=color, alpha=alpha,
            )


def _draw_arrow(
    ax: plt.Axes,
    from_pos: tuple[int, int],
    to_pos: tuple[int, int],
    color: str = "#FF6F00",
) -> None:
    """Draw a movement arrow."""
    r0, c0 = from_pos
    r1, c1 = to_pos
    if (r0, c0) == (r1, c1):
        return
    ax.annotate(
        "",
        xy=(c1, r1),
        xytext=(c0, r0),
        arrowprops={
            "arrowstyle": "->,head_width=0.3,head_length=0.2",
            "color": color,
            "lw": 2.5,
            "connectionstyle": "arc3,rad=0.15",
        },
    )


def visualize_rule_examples(output_path: Path | None = None) -> None:
    """Generate a figure showing example events for each rule.

    Creates a 5-row figure (one per rule), each showing before → after.
    """
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    fig.suptitle(
        "Micro-World: Five Physical Rules",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # ── Gravity example ──
    ax_before, ax_after = axes[0]
    _draw_grid(ax_before, "GRAVITY — Before")
    _draw_grid(ax_after, "GRAVITY — After")

    _draw_object(ax_before, "ball", "ball", 3, 2)
    _draw_object(ax_before, "table", "table", 0, 4)
    _draw_object(ax_before, "cup", "cup", 1, 4)

    _draw_object(ax_after, "ball", "ball", 0, 2)
    _draw_object(ax_after, "table", "table", 0, 4)
    _draw_object(ax_after, "cup", "cup", 1, 4)

    _draw_arrow(ax_after, (3, 2), (0, 2), "#E53935")
    ax_after.text(
        2.5, 1.5, "falls!\n(unsupported)",
        fontsize=8, ha="center", color="#E53935", style="italic",
    )
    ax_after.text(
        4, 2.0, "stays\n(on table)",
        fontsize=8, ha="center", color="#1E88E5", style="italic",
    )

    # ── Containment example ──
    ax_before, ax_after = axes[1]
    _draw_grid(ax_before, "CONTAINMENT — Before")
    _draw_grid(ax_after, "CONTAINMENT — After (push box right)")

    _draw_object(ax_before, "box", "box", 0, 1)
    _draw_object(ax_before, "ball", "ball", 0, 1, alpha=0.8)
    ax_before.text(
        1, -0.15, "(ball inside box)",
        fontsize=7, ha="center", color="#999", style="italic",
    )

    _draw_object(ax_after, "box", "box", 0, 2)
    _draw_object(ax_after, "ball", "ball", 0, 2, alpha=0.8)

    _draw_arrow(ax_after, (0, 1), (0, 2), "#43A047")
    ax_after.text(
        1.5, 0.5, "box pushed \u2192\nball follows!",
        fontsize=8, ha="center", color="#43A047", style="italic",
    )

    # ── Contact example ──
    ax_before, ax_after = axes[2]
    _draw_grid(ax_before, "CONTACT — Before")
    _draw_grid(ax_after, "CONTACT — After (push box right)")

    _draw_object(ax_before, "table", "table", 0, 2)
    _draw_object(ax_before, "box", "box", 1, 2)

    _draw_object(ax_after, "table", "table", 0, 2)
    _draw_object(ax_after, "box", "box", 1, 3)

    _draw_arrow(ax_after, (1, 2), (1, 3), "#FF6F00")
    ax_after.text(
        2.5, 1.7, "pushed \u2192",
        fontsize=8, ha="center", color="#FF6F00", style="italic",
    )

    # ── Bounce example ──
    ax_before, ax_after = axes[3]
    _draw_grid(ax_before, "BOUNCE — Before")
    _draw_grid(ax_after, "BOUNCE — After (ball falls, then bounces)")

    _draw_object(ax_before, "ball", "ball", 3, 2)

    # Ball falls to floor (row 0), then bounces up 1 (row 1)
    _draw_object(ax_after, "ball", "ball", 1, 2)

    _draw_arrow(ax_after, (3, 2), (0, 2), "#E53935")
    _draw_arrow(ax_after, (0, 2), (1, 2), "#AB47BC")
    ax_after.text(
        1.5, 1.5, "falls to floor",
        fontsize=8, ha="center", color="#E53935", style="italic",
    )
    ax_after.text(
        3.0, 0.5, "bounces up!\n(elastic)",
        fontsize=8, ha="center", color="#AB47BC", style="italic",
    )
    ax_after.text(
        2, 3.5, "only balls bounce",
        fontsize=7, ha="center", color="#777", style="italic",
    )

    # ── Breakage example ──
    ax_before, ax_after = axes[4]
    _draw_grid(ax_before, "BREAKAGE — Before")
    _draw_grid(ax_after, "BREAKAGE — After (cup falls \u2265 2 rows)")

    _draw_object(ax_before, "cup", "cup", 3, 2)

    # Cup falls from row 3 to row 0 (distance 3 >= threshold 2) -> breaks
    _draw_object(ax_after, "cup", "cup", 0, 2, alpha=0.5)

    _draw_arrow(ax_after, (3, 2), (0, 2), "#E53935")
    ax_after.text(
        1.5, 1.5, "falls 3 rows",
        fontsize=8, ha="center", color="#E53935", style="italic",
    )
    ax_after.text(
        3.0, 0.5, "BROKEN!\n(fall \u2265 2)",
        fontsize=8, ha="center", color="#C62828", fontweight="bold",
        style="italic",
    )
    ax_after.text(
        2, 3.5, "only cups break",
        fontsize=7, ha="center", color="#777", style="italic",
    )
    # Draw an X over the broken cup
    ax_after.plot(
        [1.7, 2.3], [-0.3, 0.3], color="#C62828", linewidth=3, alpha=0.7,
    )
    ax_after.plot(
        [1.7, 2.3], [0.3, -0.3], color="#C62828", linewidth=3, alpha=0.7,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_event_distribution(output_path: Path | None = None) -> None:
    """Show the distribution of generated events across rules."""
    rules = ["gravity", "containment", "contact", "bounce", "breakage"]
    colors = ["#E53935", "#43A047", "#1E88E5", "#AB47BC", "#C62828"]

    fig, axes = plt.subplots(1, len(rules), figsize=(18, 4))
    fig.suptitle(
        "Event Generation: 500 Events Per Rule",
        fontsize=13, fontweight="bold",
    )

    for ax, rule, color in zip(
        axes,
        rules,
        colors,
        strict=True,
    ):
        events = generate_rule_events(rule, n_events=500, seed=42)

        # Count events where position changed vs stayed
        moved = sum(1 for e in events if e.pos_before != e.pos_after)
        stayed = len(events) - moved

        bars = ax.bar(
            ["moved", "stayed"], [moved, stayed],
            color=[color, "#BDBDBD"], edgecolor="black", linewidth=0.5,
        )
        ax.set_title(f"{rule.upper()}", fontsize=11, fontweight="bold")
        ax.set_ylabel("count", fontsize=9)

        for bar, val in zip(bars, [moved, stayed], strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", fontsize=10, fontweight="bold",
            )

        # Show position heatmaps
        ax2 = ax.inset_axes([0.55, 0.45, 0.4, 0.4])
        heatmap = np.zeros((GRID_SIZE, GRID_SIZE))
        for e in events:
            r, c = e.pos_before
            heatmap[r, c] += 1
        ax2.imshow(
            heatmap, cmap="YlOrRd", origin="lower", aspect="equal",
            interpolation="nearest",
        )
        ax2.set_title("start positions", fontsize=7)
        ax2.set_xticks([])
        ax2.set_yticks([])

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    """Generate all world visualizations."""
    results_dir = Path(__file__).parent / "results"

    print("Generating micro-world visualizations...")
    visualize_rule_examples(results_dir / "world_rules.png")
    visualize_event_distribution(results_dir / "event_distribution.png")
    print("Done!")


if __name__ == "__main__":
    main()
