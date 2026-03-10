"""Event-to-vector encoding for causal dictionary learning.

Converts Event dataclass instances into flat binary numpy vectors
suitable for sparse dictionary learning. Each event is encoded as a
62-dimensional binary vector with exactly 5 ones (one per field).

Encoding layout (62 dimensions total):
  - obj_type:     5 dims  (one-hot over object types)
  - pos_before:  25 dims  (one-hot over 5x5 grid, row*5+col)
  - pos_after:   25 dims  (one-hot over 5x5 grid, row*5+col)
  - action:       4 dims  (one-hot over action types)
  - state_change: 3 dims  (one-hot over state change types)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.causal_dictionaries.micro_world import Event

OBJ_TYPES: list[str] = ["ball", "cup", "box", "shelf", "table"]
ACTIONS: list[str] = ["gravity_fall", "contained_move", "push", "none"]
STATE_CHANGES: list[str] = ["intact", "broken", "unchanged"]
GRID_SIZE: int = 5
VECTOR_DIM: int = 62  # 5 + 25 + 25 + 4 + 3

# Precompute field offsets for the flat vector layout.
_OFFSET_OBJ_TYPE: int = 0
_OFFSET_POS_BEFORE: int = 5
_OFFSET_POS_AFTER: int = 30
_OFFSET_ACTION: int = 55
_OFFSET_STATE_CHANGE: int = 59


def encode_event(event: Event) -> np.ndarray:
    """Encode a single Event into a flat binary vector.

    Args:
        event: An Event dataclass instance from the micro-world simulator.

    Returns:
        A (62,) float64 numpy array with exactly 5 ones and 57 zeros.

    Raises:
        ValueError: If any event field contains an unknown category.
    """
    vec = np.zeros(VECTOR_DIM, dtype=np.float64)

    # obj_type: one-hot at offset 0
    vec[_OFFSET_OBJ_TYPE + _lookup(OBJ_TYPES, event.obj_type, "obj_type")] = 1.0

    # pos_before: one-hot at offset 5 (row * 5 + col)
    row_b, col_b = event.pos_before
    vec[_OFFSET_POS_BEFORE + row_b * GRID_SIZE + col_b] = 1.0

    # pos_after: one-hot at offset 30 (row * 5 + col)
    row_a, col_a = event.pos_after
    vec[_OFFSET_POS_AFTER + row_a * GRID_SIZE + col_a] = 1.0

    # action: one-hot at offset 55
    vec[_OFFSET_ACTION + _lookup(ACTIONS, event.action, "action")] = 1.0

    # state_change: one-hot at offset 59
    vec[
        _OFFSET_STATE_CHANGE
        + _lookup(STATE_CHANGES, event.state_change, "state_change")
    ] = 1.0

    return vec


def encode_events(events: list[Event]) -> np.ndarray:
    """Batch-encode a list of Events into a binary matrix.

    Args:
        events: List of Event dataclass instances.

    Returns:
        An (N, 62) float64 numpy array where N is the number of events.
        Each row has exactly 5 ones and 57 zeros.
    """
    n = len(events)
    matrix = np.zeros((n, VECTOR_DIM), dtype=np.float64)
    for i, event in enumerate(events):
        matrix[i] = encode_event(event)
    return matrix


def encode_event_enriched(event: Event) -> np.ndarray:
    """Encode an event with displacement and height features.

    Adds continuous features that directly capture causal structure:
    displacement, magnitude, height. Total: 67 dims.

    Args:
        event: An Event dataclass instance.

    Returns:
        A (67,) float64 numpy array.
    """
    base = encode_event(event)
    r_b, c_b = event.pos_before
    r_a, c_a = event.pos_after
    dr = (r_a - r_b) / max(GRID_SIZE - 1, 1)
    dc = (c_a - c_b) / max(GRID_SIZE - 1, 1)
    mag = np.sqrt(dr**2 + dc**2)
    changed = float(event.pos_before != event.pos_after)
    height = r_b / max(GRID_SIZE - 1, 1)
    return np.concatenate([base, [dr, dc, mag, changed, height]])


VECTOR_DIM_ENRICHED: int = VECTOR_DIM + 5  # 67


def encode_event_compact(event: Event) -> np.ndarray:
    """Compact encoding: continuous positions + displacement.

    Replaces 50-dim one-hot positions with 7 continuous dims.
    Total: 19 dims. Causal features dominate.

    Args:
        event: An Event dataclass instance.

    Returns:
        A (19,) float64 numpy array.
    """
    vec = np.zeros(19, dtype=np.float64)
    # obj_type one-hot: [0:5]
    vec[_lookup(OBJ_TYPES, event.obj_type, "obj_type")] = 1.0
    r_b, c_b = event.pos_before
    r_a, c_a = event.pos_after
    scale = max(GRID_SIZE - 1, 1)
    # continuous position features: [5:12]
    vec[5] = r_b / scale
    vec[6] = c_b / scale
    vec[7] = (r_a - r_b) / scale  # displacement row
    vec[8] = (c_a - c_b) / scale  # displacement col
    vec[9] = np.sqrt(vec[7] ** 2 + vec[8] ** 2)  # magnitude
    vec[10] = float(event.pos_before != event.pos_after)  # changed
    vec[11] = r_b / scale  # height (= pos_before_row, explicit for gravity)
    # action one-hot: [12:16]
    vec[12 + _lookup(ACTIONS, event.action, "action")] = 1.0
    # state_change one-hot: [16:19]
    vec[16 + _lookup(STATE_CHANGES, event.state_change, "state_change")] = 1.0
    return vec


VECTOR_DIM_COMPACT: int = 19


def encode_event_raw(event: Event) -> np.ndarray:
    """Raw encoding: event fields as-is, no derived features.

    Uses only the fields present in the Event dataclass with minimal
    transformation (one-hot for categoricals, normalized scalars for
    positions). No displacement, magnitude, height, or changed flag —
    the model must discover these relationships itself.

    Layout (16 dimensions):
      - obj_type:     5 dims  (one-hot)
      - pos_before:   2 dims  (normalized row, col)
      - pos_after:    2 dims  (normalized row, col)
      - action:       4 dims  (one-hot)
      - state_change: 3 dims  (one-hot)

    Args:
        event: An Event dataclass instance.

    Returns:
        A (16,) float64 numpy array.
    """
    vec = np.zeros(VECTOR_DIM_RAW, dtype=np.float64)
    # obj_type one-hot: [0:5]
    vec[_lookup(OBJ_TYPES, event.obj_type, "obj_type")] = 1.0
    r_b, c_b = event.pos_before
    r_a, c_a = event.pos_after
    scale = max(GRID_SIZE - 1, 1)
    # raw positions: [5:9]
    vec[5] = r_b / scale
    vec[6] = c_b / scale
    vec[7] = r_a / scale
    vec[8] = c_a / scale
    # action one-hot: [9:13]
    vec[9 + _lookup(ACTIONS, event.action, "action")] = 1.0
    # state_change one-hot: [13:16]
    vec[13 + _lookup(STATE_CHANGES, event.state_change, "state_change")] = 1.0
    return vec


VECTOR_DIM_RAW: int = 16


def encode_events_v2(
    events: list[Event],
    encoding: str = "original",
) -> np.ndarray:
    """Batch-encode events with selectable encoding scheme.

    Args:
        events: List of Event instances.
        encoding: One of "original", "enriched", "compact", "raw".

    Returns:
        Encoded matrix of shape (N, dim).
    """
    if encoding == "original":
        return encode_events(events)
    elif encoding == "enriched":
        return np.array([encode_event_enriched(e) for e in events])
    elif encoding == "compact":
        return np.array([encode_event_compact(e) for e in events])
    elif encoding == "raw":
        return np.array([encode_event_raw(e) for e in events])
    else:
        msg = f"Unknown encoding: {encoding!r}"
        raise ValueError(msg)


def _lookup(categories: list[str], value: str, field_name: str) -> int:
    """Look up the index of a value in a category list.

    Args:
        categories: Ordered list of valid category strings.
        value: The value to look up.
        field_name: Name of the field (for error messages).

    Returns:
        Zero-based index of the value in the category list.

    Raises:
        ValueError: If the value is not found in the category list.
    """
    try:
        return categories.index(value)
    except ValueError:
        msg = (
            f"Unknown {field_name}: {value!r}. "
            f"Expected one of {categories}"
        )
        raise ValueError(msg) from None
