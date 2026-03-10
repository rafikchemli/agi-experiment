"""Tests for the causal dictionary micro-world and sparse coding."""

import numpy as np
import pytest


class TestGridWorld:
    """Test the 5x5 grid world simulator."""

    def test_create_world_with_objects(self) -> None:
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        world.place("ball", row=3, col=2)
        world.place("table", row=1, col=2)

        assert world.get_position("ball") == (3, 2)
        assert world.get_position("table") == (1, 2)

    def test_gravity_unsupported_object_falls(self) -> None:
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        world.place("ball", row=3, col=2)
        events = world.step()

        assert len(events) >= 1
        fall_event = events[0]
        assert fall_event.obj_name == "ball"
        assert fall_event.pos_before == (3, 2)
        assert fall_event.pos_after == (0, 2)
        assert fall_event.rule == "gravity"

    def test_gravity_supported_object_stays(self) -> None:
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        world.place("table", row=0, col=2)
        world.place("ball", row=1, col=2)
        events = world.step()

        fall_events = [e for e in events if e.rule == "gravity" and e.pos_before != e.pos_after]
        assert len(fall_events) == 0

    def test_containment_contents_move_with_container(self) -> None:
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        world.place("box", row=1, col=2)
        world.place("ball", row=1, col=2, inside="box")
        world.push("box", direction="right")
        events = world.step()

        box_events = [e for e in events if e.obj_name == "box"]
        ball_events = [e for e in events if e.obj_name == "ball"]
        assert len(box_events) >= 1
        assert len(ball_events) >= 1
        assert ball_events[0].pos_after == box_events[0].pos_after

    def test_contact_push_moves_object(self) -> None:
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        world.place("table", row=0, col=2)
        world.place("ball", row=1, col=2)
        world.push("ball", direction="right")
        events = world.step()

        push_events = [e for e in events if e.rule == "contact"]
        assert len(push_events) >= 1
        assert push_events[0].pos_after == (1, 3)

    def test_push_left(self) -> None:
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        world.place("ball", row=0, col=3)
        world.push("ball", direction="left")
        events = world.step()

        push_events = [e for e in events if e.rule == "contact"]
        assert len(push_events) >= 1
        assert push_events[0].pos_after == (0, 2)

    def test_push_clamps_at_grid_boundary(self) -> None:
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        world.place("ball", row=0, col=0)
        world.push("ball", direction="left")
        events = world.step()

        push_events = [e for e in events if e.rule == "contact"]
        assert len(push_events) >= 1
        # Should stay at col=0 (clamped)
        assert push_events[0].pos_after[1] == 0

    def test_gravity_falls_to_nearest_surface(self) -> None:
        """Object should fall to the top of the nearest object below, not floor."""
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        world.place("table", row=0, col=2)
        world.place("ball", row=4, col=2)
        events = world.step()

        fall_events = [e for e in events if e.rule == "gravity" and e.obj_name == "ball"]
        assert len(fall_events) == 1
        # Ball should land on top of the table (row 1), not the floor
        assert fall_events[0].pos_after == (1, 2)

    def test_object_on_floor_stays(self) -> None:
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        world.place("ball", row=0, col=2)
        events = world.step()

        fall_events = [e for e in events if e.rule == "gravity" and e.pos_before != e.pos_after]
        assert len(fall_events) == 0

    def test_get_position_unknown_object_raises(self) -> None:
        from experiments.causal_dictionaries.micro_world import GridWorld

        world = GridWorld(seed=42)
        with pytest.raises(KeyError):
            world.get_position("nonexistent")


class TestEventGeneration:
    """Test bulk event generation for training data."""

    def test_generate_gravity_events(self) -> None:
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )

        events = generate_rule_events("gravity", n_events=100, seed=42)
        assert len(events) == 100
        for e in events:
            assert e.rule == "gravity"

    def test_generate_containment_events(self) -> None:
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )

        events = generate_rule_events("containment", n_events=100, seed=42)
        assert len(events) == 100
        for e in events:
            assert e.rule == "containment"

    def test_generate_contact_events(self) -> None:
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )

        events = generate_rule_events("contact", n_events=100, seed=42)
        assert len(events) == 100
        for e in events:
            assert e.rule == "contact"

    def test_events_are_deterministic_with_seed(self) -> None:
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )

        events_a = generate_rule_events("gravity", n_events=50, seed=123)
        events_b = generate_rule_events("gravity", n_events=50, seed=123)
        for a, b in zip(events_a, events_b, strict=True):
            assert a.pos_before == b.pos_before
            assert a.pos_after == b.pos_after

    def test_gravity_events_include_negatives(self) -> None:
        """Gravity generation should include objects that don't fall."""
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )

        events = generate_rule_events("gravity", n_events=200, seed=42)
        positive = [e for e in events if e.action == "gravity_fall"]
        negative = [e for e in events if e.action == "none"]
        assert len(positive) > 0, "Should have positive gravity events"
        assert len(negative) > 0, "Should have negative gravity events"

    def test_generate_invalid_rule_raises(self) -> None:
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )

        with pytest.raises(ValueError, match="Unknown rule"):
            generate_rule_events("teleportation", n_events=10, seed=42)


class TestEventEncoding:
    """Test event -> vector encoding."""

    def test_encode_single_event(self) -> None:
        from experiments.causal_dictionaries.event_encoding import (
            encode_event,
        )
        from experiments.causal_dictionaries.micro_world import Event

        event = Event(
            obj_name="ball",
            obj_type="ball",
            pos_before=(3, 2),
            pos_after=(0, 2),
            rule="gravity",
            action="gravity_fall",
            state_change="unchanged",
        )
        vec = encode_event(event)
        assert vec.shape == (62,)
        assert vec.dtype == np.float64
        assert set(np.unique(vec)).issubset({0.0, 1.0})
        assert vec.sum() == 5.0

    def test_encode_batch(self) -> None:
        from experiments.causal_dictionaries.event_encoding import (
            encode_events,
        )
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )

        events = generate_rule_events("gravity", n_events=50, seed=42)
        matrix = encode_events(events)
        assert matrix.shape == (50, 62)
        assert np.all(matrix.sum(axis=1) == 5.0)

    def test_different_events_produce_different_vectors(self) -> None:
        from experiments.causal_dictionaries.event_encoding import (
            encode_event,
        )
        from experiments.causal_dictionaries.micro_world import Event

        e1 = Event(
            "ball",
            "ball",
            (3, 2),
            (0, 2),
            "gravity",
            "gravity_fall",
            "unchanged",
        )
        e2 = Event(
            "cup",
            "cup",
            (2, 1),
            (0, 1),
            "gravity",
            "gravity_fall",
            "unchanged",
        )
        v1 = encode_event(e1)
        v2 = encode_event(e2)
        assert not np.array_equal(v1, v2)


class TestSparseDictionary:
    """Test sparse dictionary learning via ISTA + Hebbian updates."""

    def test_dictionary_trains_and_reduces_error(self) -> None:
        from experiments.causal_dictionaries.event_encoding import (
            encode_events,
        )
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )
        from experiments.causal_dictionaries.sparse_dictionary import (
            SparseDictionary,
        )

        events = generate_rule_events("gravity", n_events=200, seed=42)
        data = encode_events(events)
        sd = SparseDictionary(n_atoms=20, seed=42)
        history = sd.train(data, epochs=10)
        assert history[-1]["loss"] < history[0]["loss"]

    def test_settle_produces_sparse_codes(self) -> None:
        from experiments.causal_dictionaries.event_encoding import (
            encode_events,
        )
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )
        from experiments.causal_dictionaries.sparse_dictionary import (
            SparseDictionary,
        )

        events = generate_rule_events("gravity", n_events=200, seed=42)
        data = encode_events(events)
        sd = SparseDictionary(n_atoms=20, seed=42)
        sd.train(data, epochs=10)
        codes = sd.encode(data[:10])
        assert codes.shape == (10, 20)
        # With 20 atoms on 62-dim binary data, expect many near-zero
        sparsity = (np.abs(codes) < 0.1).mean()
        assert sparsity > 0.3

    def test_reconstruction_reasonable(self) -> None:
        from experiments.causal_dictionaries.event_encoding import (
            encode_events,
        )
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )
        from experiments.causal_dictionaries.sparse_dictionary import (
            SparseDictionary,
        )

        events = generate_rule_events("gravity", n_events=500, seed=42)
        data = encode_events(events)
        sd = SparseDictionary(n_atoms=30, seed=42)
        sd.train(data, epochs=20)
        errors = sd.reconstruction_error(data[:50])
        assert errors.mean() < 0.5

    def test_dictionary_shape(self) -> None:
        from experiments.causal_dictionaries.event_encoding import (
            encode_events,
        )
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )
        from experiments.causal_dictionaries.sparse_dictionary import (
            SparseDictionary,
        )

        events = generate_rule_events("gravity", n_events=100, seed=42)
        data = encode_events(events)
        sd = SparseDictionary(n_atoms=25, seed=42)
        sd.train(data, epochs=5)
        assert sd.dictionary.shape == (62, 25)


class TestAnalysis:
    """Test analysis functions for atom-rule affinity and specialization."""

    def _train_dictionary(self) -> tuple:  # type: ignore[type-arg]
        from experiments.causal_dictionaries.event_encoding import (
            encode_events,
        )
        from experiments.causal_dictionaries.micro_world import (
            generate_rule_events,
        )
        from experiments.causal_dictionaries.sparse_dictionary import (
            SparseDictionary,
        )

        grav = encode_events(generate_rule_events("gravity", n_events=200, seed=42))
        cont = encode_events(generate_rule_events("containment", n_events=200, seed=43))
        ctct = encode_events(generate_rule_events("contact", n_events=200, seed=44))
        all_data = np.vstack([grav, cont, ctct])
        sd = SparseDictionary(n_atoms=30, seed=42)
        sd.train(all_data, epochs=15)
        return sd, grav, cont, ctct

    def test_atom_rule_affinity_matrix(self) -> None:
        from experiments.causal_dictionaries.analysis import (
            atom_rule_affinity,
        )

        sd, grav, cont, ctct = self._train_dictionary()
        affinity = atom_rule_affinity(
            sd,
            {"gravity": grav, "containment": cont, "contact": ctct},
        )
        assert affinity.shape == (30, 3)
        assert np.all(affinity >= 0)

    def test_specialization_score(self) -> None:
        from experiments.causal_dictionaries.analysis import (
            specialization_scores,
        )

        sd, grav, cont, ctct = self._train_dictionary()
        scores = specialization_scores(
            sd,
            {"gravity": grav, "containment": cont, "contact": ctct},
        )
        assert scores.shape == (30,)
        assert np.all(scores >= 1.0 / 3.0 - 0.01)
        assert np.all(scores <= 1.0 + 0.01)


class TestCompositionEvents:
    """Test composition event generation for multi-rule scenarios."""

    def test_gravity_containment_composition(self) -> None:
        from experiments.causal_dictionaries.micro_world import (
            generate_composition_events,
        )

        events = generate_composition_events(["gravity", "containment"], n_events=50, seed=42)
        assert len(events) == 50

    def test_all_three_composition(self) -> None:
        from experiments.causal_dictionaries.micro_world import (
            generate_composition_events,
        )

        events = generate_composition_events(
            ["gravity", "containment", "contact"],
            n_events=50,
            seed=42,
        )
        assert len(events) == 50
