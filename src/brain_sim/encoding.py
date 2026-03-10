"""Retinal encoding: images to spike-driving currents.

In biology, the retina preprocesses visual input before it reaches
the cortex. Retinal ganglion cells have center-surround receptive
fields that enhance edges and reduce redundancy.

For our initial experiments, we use simple rate coding:
pixel intensity -> input current -> firing rate.

This tests Assumption A2 (rate coding is sufficient).
If it fails, we add center-surround filtering.
"""

import numpy as np


def generate_horizontal_bar(
    grid_size: int = 8,
    row: int | None = None,
    width: int = 1,
) -> np.ndarray:
    """Generate a binary image with a horizontal bar.

    Args:
        grid_size: Image size (grid_size x grid_size).
        row: Which row to place the bar (default: center).
        width: Bar width in pixels.

    Returns:
        2D float array with 1.0 for bar pixels, 0.0 elsewhere.
    """
    if row is None:
        row = grid_size // 2
    img = np.zeros((grid_size, grid_size), dtype=np.float64)
    for r in range(max(0, row), min(grid_size, row + width)):
        img[r, :] = 1.0
    return img


def generate_vertical_bar(
    grid_size: int = 8,
    col: int | None = None,
    width: int = 1,
) -> np.ndarray:
    """Generate a binary image with a vertical bar.

    Args:
        grid_size: Image size (grid_size x grid_size).
        col: Which column to place the bar (default: center).
        width: Bar width in pixels.

    Returns:
        2D float array with 1.0 for bar pixels, 0.0 elsewhere.
    """
    if col is None:
        col = grid_size // 2
    img = np.zeros((grid_size, grid_size), dtype=np.float64)
    for c in range(max(0, col), min(grid_size, col + width)):
        img[:, c] = 1.0
    return img


def generate_diagonal_bar(
    grid_size: int = 8,
    direction: str = "right",
    width: int = 2,
) -> np.ndarray:
    """Generate a binary image with a diagonal bar.

    Args:
        grid_size: Image size (grid_size x grid_size).
        direction: "right" for \\ diagonal, "left" for / diagonal.
        width: Bar width (fills adjacent columns for visibility).

    Returns:
        2D float array with 1.0 for bar pixels, 0.0 elsewhere.
    """
    img = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(grid_size):
        base_col = i if direction == "right" else grid_size - 1 - i
        for w in range(width):
            col = base_col + w
            if 0 <= col < grid_size:
                img[i, col] = 1.0
    return img


def image_to_currents(
    image: np.ndarray,
    max_current: float = 10.0,
    base_id: int = 0,
) -> dict[int, float]:
    """Convert a 2D image to neuron input currents (rate coding).

    Each pixel maps to one neuron. Pixel value [0, 1] maps to
    current [0, max_current]. Higher current = higher firing rate.

    This is Assumption A2: rate coding is sufficient.

    Args:
        image: 2D array with values in [0, 1].
        max_current: Maximum current for brightest pixel.
        base_id: Starting neuron ID for the image pixels.

    Returns:
        Dict of neuron_id -> current.
    """
    flat = image.flatten()
    return {base_id + i: float(v * max_current) for i, v in enumerate(flat)}
