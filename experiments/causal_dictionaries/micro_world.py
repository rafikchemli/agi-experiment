"""5x5 grid world simulator with physical rules for causal dictionary learning.

Implements a minimal physics engine with five rules:
  - Gravity: unsupported objects fall to the nearest surface below.
  - Containment: objects inside a container share its position.
  - Contact: pushing an object moves it one cell left or right.
  - Bounce: elastic objects (ball) bounce 1 row up after a gravity fall.
  - Breakage: fragile objects (cup) break when falling >= 2 rows.

The simulator generates transition events that serve as training data
for sparse dictionary learning of causal primitives.
"""

from dataclasses import dataclass, field

import numpy as np

GRID_ROWS = 5
GRID_COLS = 5
OBJECT_TYPES = ("ball", "cup", "box", "shelf", "table")

# Object type properties for bounce and breakage rules
ELASTIC_TYPES = ("ball",)       # Bounce after gravity fall
FRAGILE_TYPES = ("cup",)        # Break on hard impact
BREAKAGE_THRESHOLD = 2          # Minimum fall distance to trigger breakage


@dataclass(frozen=True)
class Event:
    """A single transition event in the micro-world.

    Attributes:
        obj_name: Name of the object (e.g., "ball").
        obj_type: Type of the object (one of OBJECT_TYPES).
        pos_before: Position before the step as (row, col).
        pos_after: Position after the step as (row, col).
        rule: Which rule generated this event ("gravity", "containment",
            "contact", or "none").
        action: Specific action taken ("gravity_fall", "contained_move",
            "push", or "none").
        state_change: Object state after the event ("intact", "broken",
            or "unchanged").
    """

    obj_name: str
    obj_type: str
    pos_before: tuple[int, int]
    pos_after: tuple[int, int]
    rule: str
    action: str
    state_change: str


@dataclass
class _ObjectState:
    """Internal mutable state for an object in the grid world.

    Attributes:
        name: Object name.
        obj_type: Object type (one of OBJECT_TYPES).
        row: Current row (0=floor, 4=top).
        col: Current column.
        inside: Name of the container this object is inside, or None.
    """

    name: str
    obj_type: str
    row: int
    col: int
    inside: str | None = None


@dataclass
class GridWorld:
    """A 5x5 grid world with gravity, containment, and contact physics.

    Args:
        seed: Random seed for reproducibility.
    """

    seed: int = 42
    _objects: dict[str, _ObjectState] = field(
        default_factory=dict, init=False
    )
    _push_queue: list[tuple[str, str]] = field(
        default_factory=list, init=False
    )
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the random number generator."""
        self._rng = np.random.default_rng(self.seed)

    def place(
        self,
        name: str,
        row: int,
        col: int,
        inside: str | None = None,
    ) -> None:
        """Place an object on the grid.

        Args:
            name: Object name (also used to infer type from OBJECT_TYPES).
            row: Row position (0=floor, 4=top).
            col: Column position.
            inside: Name of the container object, or None.
        """
        obj_type = _infer_type(name)
        self._objects[name] = _ObjectState(
            name=name, obj_type=obj_type, row=row, col=col, inside=inside
        )

    def push(self, name: str, direction: str) -> None:
        """Queue a push action for the next step.

        Args:
            name: Name of the object to push.
            direction: Direction to push ("left" or "right").
        """
        self._push_queue.append((name, direction))

    def get_position(self, name: str) -> tuple[int, int]:
        """Get the current position of an object.

        Args:
            name: Object name.

        Returns:
            Position as (row, col).

        Raises:
            KeyError: If the object does not exist.
        """
        if name not in self._objects:
            msg = f"Object '{name}' not found"
            raise KeyError(msg)
        obj = self._objects[name]
        return (obj.row, obj.col)

    def step(self) -> list[Event]:
        """Apply all physics rules and return transition events.

        Execution order:
          1. Contact (process push queue)
          2. Containment (sync contained objects to containers)
          3. Gravity (unsupported objects fall)
          4. Bounce (elastic objects bounce up 1 row after falling)
          5. Breakage (fragile objects break on hard landing)

        Returns:
            List of events generated during this step.
        """
        events: list[Event] = []

        # Record positions before any physics
        pos_before: dict[str, tuple[int, int]] = {
            name: (obj.row, obj.col)
            for name, obj in self._objects.items()
        }

        # --- 1. Contact: process push queue ---
        pushed_names: set[str] = set()
        for name, direction in self._push_queue:
            if name not in self._objects:
                continue
            obj = self._objects[name]
            old_pos = (obj.row, obj.col)
            delta = 1 if direction == "right" else -1
            new_col = max(0, min(GRID_COLS - 1, obj.col + delta))
            obj.col = new_col
            pushed_names.add(name)

            events.append(Event(
                obj_name=name,
                obj_type=obj.obj_type,
                pos_before=old_pos,
                pos_after=(obj.row, obj.col),
                rule="contact",
                action="push",
                state_change="unchanged",
            ))

        self._push_queue.clear()

        # --- 2. Containment: sync contained objects to containers ---
        for obj in self._objects.values():
            if obj.inside is not None and obj.inside in self._objects:
                container = self._objects[obj.inside]
                old_pos = pos_before[obj.name]
                obj.row = container.row
                obj.col = container.col
                new_pos = (obj.row, obj.col)

                if old_pos != new_pos:
                    events.append(Event(
                        obj_name=obj.name,
                        obj_type=obj.obj_type,
                        pos_before=old_pos,
                        pos_after=new_pos,
                        rule="containment",
                        action="contained_move",
                        state_change="unchanged",
                    ))

        # --- 3. Gravity: unsupported objects fall ---
        # Track falls for bounce/breakage rules
        gravity_falls: dict[str, int] = {}  # name → fall distance

        # Sort by row ascending so lower objects settle first
        sorted_objs = sorted(
            self._objects.values(),
            key=lambda o: o.row,
        )

        for obj in sorted_objs:
            # Contained objects are handled by containment rule
            if obj.inside is not None:
                continue

            old_pos = (obj.row, obj.col)

            if obj.row == 0:
                # Already on floor -- no gravity event with movement
                events.append(Event(
                    obj_name=obj.name,
                    obj_type=obj.obj_type,
                    pos_before=old_pos,
                    pos_after=old_pos,
                    rule="gravity",
                    action="none",
                    state_change="unchanged",
                ))
                continue

            # Check if supported: is there an object at (row-1, same col)?
            supported = self._is_supported(obj)

            if supported:
                events.append(Event(
                    obj_name=obj.name,
                    obj_type=obj.obj_type,
                    pos_before=old_pos,
                    pos_after=old_pos,
                    rule="gravity",
                    action="none",
                    state_change="unchanged",
                ))
            else:
                # Fall to the nearest surface below
                landing_row = self._find_landing_row(obj)
                fall_distance = obj.row - landing_row
                obj.row = landing_row
                new_pos = (obj.row, obj.col)
                gravity_falls[obj.name] = fall_distance

                events.append(Event(
                    obj_name=obj.name,
                    obj_type=obj.obj_type,
                    pos_before=old_pos,
                    pos_after=new_pos,
                    rule="gravity",
                    action="gravity_fall",
                    state_change="unchanged",
                ))

        # --- 4. Bounce: elastic objects bounce up 1 row after falling ---
        for name, fall_dist in gravity_falls.items():
            obj = self._objects[name]
            if obj.obj_type in ELASTIC_TYPES and obj.row < GRID_ROWS - 1:
                bounce_before = (obj.row, obj.col)
                obj.row += 1
                events.append(Event(
                    obj_name=name,
                    obj_type=obj.obj_type,
                    pos_before=bounce_before,
                    pos_after=(obj.row, obj.col),
                    rule="bounce",
                    action="bounce",
                    state_change="unchanged",
                ))

        # --- 5. Breakage: fragile objects break on hard landing ---
        for name, fall_dist in gravity_falls.items():
            obj = self._objects[name]
            if (
                obj.obj_type in FRAGILE_TYPES
                and fall_dist >= BREAKAGE_THRESHOLD
            ):
                events.append(Event(
                    obj_name=name,
                    obj_type=obj.obj_type,
                    pos_before=(obj.row, obj.col),
                    pos_after=(obj.row, obj.col),
                    rule="breakage",
                    action="break_on_impact",
                    state_change="broken",
                ))

        return events

    def _is_supported(self, obj: _ObjectState) -> bool:
        """Check if an object is supported by something below it.

        Args:
            obj: The object to check.

        Returns:
            True if the object is on the floor or has support at (row-1, col).
        """
        if obj.row == 0:
            return True
        for other in self._objects.values():
            if other.name == obj.name:
                continue
            if other.row == obj.row - 1 and other.col == obj.col:
                return True
        return False

    def _find_landing_row(self, obj: _ObjectState) -> int:
        """Find the row an unsupported object falls to.

        Args:
            obj: The falling object.

        Returns:
            The row the object lands on (on top of the highest
            object below it, or row 0 for the floor).
        """
        highest_below = -1
        for other in self._objects.values():
            if other.name == obj.name:
                continue
            if other.col == obj.col and other.row < obj.row:
                highest_below = max(highest_below, other.row)

        if highest_below == -1:
            return 0
        return highest_below + 1


def _infer_type(name: str) -> str:
    """Infer object type from its name.

    Strips trailing digits and checks against known types. Falls back
    to the raw name if no known type prefix is found.

    Args:
        name: Object name (e.g., "ball", "cup2", "table_1").

    Returns:
        The inferred object type string.
    """
    base = name.rstrip("0123456789").rstrip("_")
    for t in OBJECT_TYPES:
        if base == t:
            return t
    return base


def generate_rule_events(
    rule: str,
    n_events: int,
    seed: int = 42,
) -> list[Event]:
    """Generate a batch of events for a specific physics rule.

    Creates diverse micro-world scenarios and collects events until
    the requested count is reached. Includes both positive examples
    (rule fires) and negative examples (rule doesn't fire).

    Args:
        rule: Which rule to generate events for
            ("gravity", "containment", or "contact").
        n_events: Exact number of events to generate.
        seed: Random seed for reproducibility.

    Returns:
        A list of exactly ``n_events`` Event objects.

    Raises:
        ValueError: If the rule is not recognized.
    """
    if rule == "gravity":
        return _generate_gravity_events(n_events, seed)
    elif rule == "containment":  # noqa: RET505
        return _generate_containment_events(n_events, seed)
    elif rule == "contact":
        return _generate_contact_events(n_events, seed)
    elif rule == "bounce":
        return _generate_bounce_events(n_events, seed)
    elif rule == "breakage":
        return _generate_breakage_events(n_events, seed)
    else:
        msg = f"Unknown rule: {rule!r}"
        raise ValueError(msg)


def _generate_gravity_events(n_events: int, seed: int) -> list[Event]:
    """Generate gravity events with both falls and stable objects.

    Args:
        n_events: Number of events to generate.
        seed: Random seed.

    Returns:
        List of gravity events.
    """
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    scenario_idx = 0

    while len(events) < n_events:
        world = GridWorld(seed=int(rng.integers(0, 2**31)))
        obj_type = OBJECT_TYPES[scenario_idx % len(OBJECT_TYPES)]
        name = f"{obj_type}{scenario_idx}"

        # Alternate between unsupported (positive) and supported (negative)
        if scenario_idx % 3 == 0:
            # Positive: place at random height with no support
            row = int(rng.integers(1, GRID_ROWS))
            col = int(rng.integers(0, GRID_COLS))
            world.place(name, row=row, col=col)
        elif scenario_idx % 3 == 1:
            # Negative: place on floor
            col = int(rng.integers(0, GRID_COLS))
            world.place(name, row=0, col=col)
        else:
            # Negative: place on top of another object
            col = int(rng.integers(0, GRID_COLS))
            support_type = OBJECT_TYPES[
                (scenario_idx + 1) % len(OBJECT_TYPES)
            ]
            support_name = f"{support_type}{scenario_idx}_support"
            world.place(support_name, row=0, col=col)
            world.place(name, row=1, col=col)

        step_events = world.step()
        for e in step_events:
            if e.obj_name == name and e.rule == "gravity":
                events.append(e)
                if len(events) >= n_events:
                    break

        scenario_idx += 1

    return events[:n_events]


def _generate_containment_events(n_events: int, seed: int) -> list[Event]:
    """Generate containment events by moving containers with objects inside.

    Args:
        n_events: Number of events to generate.
        seed: Random seed.

    Returns:
        List of containment events.
    """
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    scenario_idx = 0
    container_types = ("box", "cup")

    while len(events) < n_events:
        world = GridWorld(seed=int(rng.integers(0, 2**31)))

        container_type = container_types[scenario_idx % len(container_types)]
        container_name = f"{container_type}{scenario_idx}"
        content_type = "ball"
        content_name = f"{content_type}{scenario_idx}"

        col = int(rng.integers(1, GRID_COLS - 1))  # Room to push
        world.place(container_name, row=0, col=col)
        world.place(
            content_name, row=0, col=col, inside=container_name
        )

        direction = "right" if rng.random() < 0.5 else "left"
        world.push(container_name, direction=direction)

        step_events = world.step()
        for e in step_events:
            if e.rule == "containment":
                events.append(e)
                if len(events) >= n_events:
                    break

        scenario_idx += 1

    return events[:n_events]


def generate_composition_events(
    rules: list[str],
    n_events: int,
    seed: int = 42,
) -> list[Event]:
    """Generate events where multiple rules interact simultaneously.

    Creates GridWorld scenarios where multiple physics rules must fire
    together, then runs step() to collect the resulting multi-rule events.

    Supported combinations:
        - ["gravity", "containment"]: object inside container loses
          support, both fall together.
        - ["gravity", "contact"]: push non-elastic object off support,
          it falls.
        - ["containment", "contact"]: push container, contents follow.
        - ["gravity", "containment", "contact"]: push container off
          edge, contents follow and fall.
        - ["gravity", "bounce"]: ball at height falls and bounces.
        - ["gravity", "breakage"]: cup at height >= 2 falls and breaks.
        - ["gravity", "contact", "bounce"]: push ball off support,
          it falls and bounces.
        - ["gravity", "contact", "breakage"]: push cup off high
          support, it falls and breaks.

    Args:
        rules: List of rule names to combine.
        n_events: Exact number of events to generate.
        seed: Random seed for reproducibility.

    Returns:
        A list of exactly ``n_events`` Event objects.
    """
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    scenario_idx = 0
    rule_set = frozenset(rules)

    while len(events) < n_events:
        world = GridWorld(seed=int(rng.integers(0, 2**31)))

        if rule_set == frozenset({"gravity", "containment"}):
            # Box with ball inside at height 2, no support -> both fall
            # Ball is contained so no bounce; box is not fragile so no break
            col = int(rng.integers(0, GRID_COLS))
            height = int(rng.integers(2, GRID_ROWS))
            box_name = f"box{scenario_idx}"
            ball_name = f"ball{scenario_idx}"
            world.place(box_name, row=height, col=col)
            world.place(
                ball_name, row=height, col=col, inside=box_name
            )

        elif rule_set == frozenset({"gravity", "contact"}):
            # Box on a table, push box off the edge so it falls
            # Box is non-elastic (no bounce) and non-fragile (no break)
            col = int(rng.integers(1, GRID_COLS - 1))
            table_name = f"table{scenario_idx}"
            box_name = f"box{scenario_idx}"
            world.place(table_name, row=0, col=col)
            world.place(box_name, row=1, col=col)
            direction = "right" if rng.random() < 0.5 else "left"
            world.push(box_name, direction=direction)

        elif rule_set == frozenset({"containment", "contact"}):
            # Push box containing ball -> ball follows
            col = int(rng.integers(1, GRID_COLS - 1))
            box_name = f"box{scenario_idx}"
            ball_name = f"ball{scenario_idx}"
            world.place(box_name, row=0, col=col)
            world.place(
                ball_name, row=0, col=col, inside=box_name
            )
            direction = "right" if rng.random() < 0.5 else "left"
            world.push(box_name, direction=direction)

        elif rule_set == frozenset(
            {"gravity", "containment", "contact"}
        ):
            # Push box-with-ball off a surface -> push + fall + follow
            # Box is non-elastic, non-fragile; ball is contained
            col = int(rng.integers(1, GRID_COLS - 1))
            table_name = f"table{scenario_idx}"
            box_name = f"box{scenario_idx}"
            ball_name = f"ball{scenario_idx}"
            world.place(table_name, row=0, col=col)
            world.place(box_name, row=1, col=col)
            world.place(
                ball_name, row=1, col=col, inside=box_name
            )
            direction = "right" if rng.random() < 0.5 else "left"
            world.push(box_name, direction=direction)

        elif rule_set == frozenset({"gravity", "bounce"}):
            # Ball at height, unsupported -> falls and bounces
            col = int(rng.integers(0, GRID_COLS))
            height = int(rng.integers(1, GRID_ROWS))
            ball_name = f"ball{scenario_idx}"
            world.place(ball_name, row=height, col=col)

        elif rule_set == frozenset({"gravity", "breakage"}):
            # Cup at height >= 2, unsupported -> falls and breaks
            col = int(rng.integers(0, GRID_COLS))
            height = int(rng.integers(2, GRID_ROWS))
            cup_name = f"cup{scenario_idx}"
            world.place(cup_name, row=height, col=col)

        elif rule_set == frozenset({"gravity", "contact", "bounce"}):
            # Ball on table, push off -> falls and bounces
            col = int(rng.integers(1, GRID_COLS - 1))
            table_name = f"table{scenario_idx}"
            ball_name = f"ball{scenario_idx}"
            world.place(table_name, row=0, col=col)
            world.place(ball_name, row=1, col=col)
            direction = "right" if rng.random() < 0.5 else "left"
            world.push(ball_name, direction=direction)

        elif rule_set == frozenset(
            {"gravity", "contact", "breakage"}
        ):
            # Cup on high stack, push off -> falls >= 2 and breaks
            col = int(rng.integers(1, GRID_COLS - 1))
            table_name = f"table{scenario_idx}"
            shelf_name = f"shelf{scenario_idx}"
            cup_name = f"cup{scenario_idx}"
            world.place(table_name, row=0, col=col)
            world.place(shelf_name, row=1, col=col)
            world.place(cup_name, row=2, col=col)
            direction = "right" if rng.random() < 0.5 else "left"
            world.push(cup_name, direction=direction)

        else:
            msg = f"Unsupported rule combination: {rules}"
            raise ValueError(msg)

        step_events = world.step()
        for e in step_events:
            events.append(e)
            if len(events) >= n_events:
                break

        scenario_idx += 1

    return events[:n_events]


def _generate_contact_events(n_events: int, seed: int) -> list[Event]:
    """Generate contact/push events.

    Args:
        n_events: Number of events to generate.
        seed: Random seed.

    Returns:
        List of contact events.
    """
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    scenario_idx = 0

    while len(events) < n_events:
        world = GridWorld(seed=int(rng.integers(0, 2**31)))
        obj_type = OBJECT_TYPES[scenario_idx % len(OBJECT_TYPES)]
        name = f"{obj_type}{scenario_idx}"

        col = int(rng.integers(0, GRID_COLS))
        world.place(name, row=0, col=col)

        direction = "right" if rng.random() < 0.5 else "left"
        world.push(name, direction=direction)

        step_events = world.step()
        for e in step_events:
            if e.rule == "contact":
                events.append(e)
                if len(events) >= n_events:
                    break

        scenario_idx += 1

    return events[:n_events]


def _generate_bounce_events(n_events: int, seed: int) -> list[Event]:
    """Generate bounce events from elastic objects falling.

    Creates ball objects at various heights and collects bounce events
    (upward displacement after landing). Only balls are elastic.

    Args:
        n_events: Number of events to generate.
        seed: Random seed.

    Returns:
        List of bounce events.
    """
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    scenario_idx = 0

    while len(events) < n_events:
        world = GridWorld(seed=int(rng.integers(0, 2**31)))
        name = f"ball{scenario_idx}"

        # Place ball at random height with no support
        row = int(rng.integers(1, GRID_ROWS))
        col = int(rng.integers(0, GRID_COLS))
        world.place(name, row=row, col=col)

        step_events = world.step()
        for e in step_events:
            if e.rule == "bounce":
                events.append(e)
                if len(events) >= n_events:
                    break

        scenario_idx += 1

    return events[:n_events]


def _generate_breakage_events(n_events: int, seed: int) -> list[Event]:
    """Generate breakage events from fragile objects falling hard.

    Creates cup objects at heights >= BREAKAGE_THRESHOLD and collects
    breakage events (state_change="broken"). Also generates negative
    examples (small falls that don't trigger breakage).

    Args:
        n_events: Number of events to generate.
        seed: Random seed.

    Returns:
        List of breakage events.
    """
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    scenario_idx = 0

    while len(events) < n_events:
        world = GridWorld(seed=int(rng.integers(0, 2**31)))
        name = f"cup{scenario_idx}"

        # Place cup at height >= 2 to ensure breakage
        row = int(rng.integers(BREAKAGE_THRESHOLD, GRID_ROWS))
        col = int(rng.integers(0, GRID_COLS))
        world.place(name, row=row, col=col)

        step_events = world.step()
        for e in step_events:
            if e.rule == "breakage":
                events.append(e)
                if len(events) >= n_events:
                    break

        scenario_idx += 1

    return events[:n_events]
