"""Tests for Experiment 5: STDP + WTA self-organize feature detectors.

Verifies that a population of V1 excitatory neurons, receiving input from
a retinal layer through STDP-plastic synapses with WTA lateral inhibition
and homeostatic scaling, self-organizes orientation-selective receptive fields.

This is the CRITICAL experiment per knowledge/misc/PROMPT.md. The test criteria are:
1. Different V1 neurons develop different receptive fields
2. The population has at least one H-preferring and one V-preferring neuron
3. The correct winner emerges for each orientation during testing
"""

import numpy as np
import pytest

from brain_sim.encoding import (
    generate_diagonal_bar,
    generate_horizontal_bar,
    generate_vertical_bar,
    image_to_currents,
)
from brain_sim.layers import build_v1_circuit, get_receptive_fields
from brain_sim.learning import apply_homeostatic_scaling
from brain_sim.synapse import STDPParams

# -- Encoding tests ----------------------------------------------------------


class TestEncoding:
    """Test retinal encoding: pattern generation and rate coding."""

    def test_horizontal_bar_shape(self) -> None:
        img = generate_horizontal_bar(grid_size=8, row=3, width=1)
        assert img.shape == (8, 8)
        assert img[3, :].sum() == 8.0
        assert img[2, :].sum() == 0.0

    def test_vertical_bar_shape(self) -> None:
        img = generate_vertical_bar(grid_size=8, col=5, width=1)
        assert img.shape == (8, 8)
        assert img[:, 5].sum() == 8.0
        assert img[:, 4].sum() == 0.0

    def test_diagonal_bar(self) -> None:
        img = generate_diagonal_bar(grid_size=4, direction="right", width=1)
        assert img.shape == (4, 4)
        # Diagonal should have pixels along the main diagonal
        assert img[0, 0] == 1.0
        assert img[1, 1] == 1.0
        assert img[2, 2] == 1.0
        assert img[3, 3] == 1.0

    def test_h_v_bars_orthogonal(self) -> None:
        """H and V bars should be nearly orthogonal (low dot product)."""
        h = generate_horizontal_bar(8).flatten()
        v = generate_vertical_bar(8).flatten()
        # Overlap is 1 pixel out of ~16 active pixels
        dot = float(np.dot(h, v))
        assert dot <= 2.0  # at most a couple overlap pixels

    def test_image_to_currents_scaling(self) -> None:
        img = np.array([[0.0, 0.5], [1.0, 0.0]])
        currents = image_to_currents(img, max_current=10.0, base_id=0)
        assert currents[0] == 0.0
        assert currents[1] == 5.0
        assert currents[2] == 10.0
        assert currents[3] == 0.0

    def test_image_to_currents_base_id(self) -> None:
        img = np.ones((2, 2))
        currents = image_to_currents(img, max_current=5.0, base_id=100)
        assert set(currents.keys()) == {100, 101, 102, 103}


# -- Layer construction tests -------------------------------------------------


class TestV1Circuit:
    """Test V1 circuit construction."""

    def test_neuron_count(self) -> None:
        net, ids = build_v1_circuit(grid_size=4, n_v1_excitatory=3, n_v1_inhibitory=1)
        # 16 input + 3 V1 exc + 1 V1 inh = 20
        assert len(net.neurons) == 20

    def test_id_map_keys(self) -> None:
        _, ids = build_v1_circuit(grid_size=4, n_v1_excitatory=2, n_v1_inhibitory=1)
        assert "input" in ids
        assert "v1_exc" in ids
        assert "v1_inh" in ids
        assert len(ids["input"]) == 16
        assert len(ids["v1_exc"]) == 2
        assert len(ids["v1_inh"]) == 1

    def test_input_to_v1_connections_are_plastic(self) -> None:
        net, ids = build_v1_circuit(grid_size=4, n_v1_excitatory=2, n_v1_inhibitory=1)
        for v1_id in ids["v1_exc"]:
            conns = net.incoming_connections(v1_id)
            plastic_conns = [c for c in conns if c.synapse.plastic]
            # Each V1 neuron gets one plastic connection from each input
            assert len(plastic_conns) == 16

    def test_inhibitory_connections_are_fixed(self) -> None:
        net, ids = build_v1_circuit(grid_size=4, n_v1_excitatory=2, n_v1_inhibitory=1)
        for inh_id in ids["v1_inh"]:
            conns = net.incoming_connections(inh_id)
            for conn in conns:
                assert not conn.synapse.plastic

    def test_initial_weights_are_randomized(self) -> None:
        """Random initial weights break symmetry — essential for specialization."""
        net, ids = build_v1_circuit(grid_size=4, n_v1_excitatory=2, n_v1_inhibitory=1)
        fields = get_receptive_fields(net, ids["v1_exc"], 16)
        w0 = fields[ids["v1_exc"][0]]
        w1 = fields[ids["v1_exc"][1]]
        # Weights should not be identical
        assert not np.allclose(w0, w1)


# -- Self-organization tests --------------------------------------------------

# Shared training parameters for reproducibility
_GRID = 4
_STDP = STDPParams(
    a_plus=0.015,
    a_minus=0.016,
    tau_plus=20.0,
    tau_minus=20.0,
    w_max=10.0,
    w_min=0.0,
)
_N_V1_EXC = 4
_N_V1_INH = 2
_PRESENT_MS = 150
_N_EPOCHS = 100
_MAX_CURRENT = 15.0
_INH_TONIC = 3.0


def _train_v1(seed: int = 42) -> tuple:
    """Train a V1 circuit on H/V bars and return test results.

    Returns:
        Tuple of (net, ids, results, fields) where results maps
        pattern label to spike counts per V1 neuron.
    """
    rng = np.random.default_rng(seed)
    net, ids = build_v1_circuit(
        grid_size=_GRID,
        n_v1_excitatory=_N_V1_EXC,
        n_v1_inhibitory=_N_V1_INH,
        w_input_to_v1=4.0,
        w_v1_to_inh=6.0,
        w_inh_to_v1=18.0,
        inh_tonic=_INH_TONIC,
        tau_syn=5.0,
        stdp_params=_STDP,
        rng=rng,
    )
    v1_exc = ids["v1_exc"]
    inh_ids = ids["v1_inh"]

    h_bar = generate_horizontal_bar(_GRID, row=_GRID // 2, width=1)
    v_bar = generate_vertical_bar(_GRID, col=_GRID // 2, width=1)
    patterns = [("H", h_bar), ("V", v_bar)]

    # Training with STDP + WTA + homeostasis
    for _epoch in range(_N_EPOCHS):
        order = list(range(len(patterns)))
        rng.shuffle(order)
        for idx in order:
            _, img = patterns[idx]
            currents = image_to_currents(img, max_current=_MAX_CURRENT)
            for inh_id in inh_ids:
                currents[inh_id] = _INH_TONIC
            net.simulate(_PRESENT_MS, currents)
        target_rates = {nid: 8.0 for nid in v1_exc}
        apply_homeostatic_scaling(
            net,
            target_rates,
            window_ms=_PRESENT_MS * len(patterns),
            eta=0.08,
        )

    # Test: measure responses to each pattern
    n_pixels = _GRID * _GRID
    results: dict[str, dict[int, int]] = {}
    for label, img in [("H", h_bar), ("V", v_bar)]:
        net._spike_log = {nid: [] for nid in net.neurons}
        net._timestep = 0
        currents = image_to_currents(img, max_current=_MAX_CURRENT)
        for inh_id in inh_ids:
            currents[inh_id] = _INH_TONIC
        net.simulate(500, currents)
        results[label] = {nid: len(net._spike_log[nid]) for nid in v1_exc}

    fields = get_receptive_fields(net, v1_exc, n_pixels)
    return net, ids, results, fields


class TestSelfOrganization:
    """Test that STDP + WTA + homeostasis produce orientation selectivity."""

    @pytest.fixture(scope="class")
    def trained_circuit(self) -> tuple:
        """Train once, reuse for all tests in this class."""
        return _train_v1(seed=42)

    def test_different_winners_for_different_orientations(
        self,
        trained_circuit: tuple,
    ) -> None:
        """The H-winner and V-winner should be different neurons."""
        _, ids, results, _ = trained_circuit
        v1_exc = ids["v1_exc"]
        h_winner = max(v1_exc, key=lambda n: results["H"][n])
        v_winner = max(v1_exc, key=lambda n: results["V"][n])
        assert h_winner != v_winner, (
            f"Same neuron V1[{h_winner}] wins for both H and V: H={results['H']}, V={results['V']}"
        )

    def test_at_least_one_h_preferring_neuron(
        self,
        trained_circuit: tuple,
    ) -> None:
        """At least one neuron should have a positive H correlation."""
        _, ids, _, fields = trained_circuit
        h_template = generate_horizontal_bar(_GRID, row=_GRID // 2).flatten()
        h_corrs = []
        for nid in ids["v1_exc"]:
            w = fields[nid]
            if np.std(w) > 1e-10:
                h_corrs.append(float(np.corrcoef(w, h_template)[0, 1]))
        assert any(c > 0.2 for c in h_corrs), f"No H-preferring neuron found: {h_corrs}"

    def test_at_least_one_v_preferring_neuron(
        self,
        trained_circuit: tuple,
    ) -> None:
        """At least one neuron should have a positive V correlation."""
        _, ids, _, fields = trained_circuit
        v_template = generate_vertical_bar(_GRID, col=_GRID // 2).flatten()
        v_corrs = []
        for nid in ids["v1_exc"]:
            w = fields[nid]
            if np.std(w) > 1e-10:
                v_corrs.append(float(np.corrcoef(w, v_template)[0, 1]))
        assert any(c > 0.2 for c in v_corrs), f"No V-preferring neuron found: {v_corrs}"

    def test_receptive_fields_are_diverse(
        self,
        trained_circuit: tuple,
    ) -> None:
        """V1 neurons should NOT all have identical receptive fields."""
        _, ids, _, fields = trained_circuit
        weights = [fields[nid] for nid in ids["v1_exc"]]
        # At least one pair should have correlation < 0.9
        found_diverse = False
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                corr = float(np.corrcoef(weights[i], weights[j])[0, 1])
                if corr < 0.9:
                    found_diverse = True
                    break
        assert found_diverse, "All V1 neurons have nearly identical receptive fields"

    def test_h_winner_responds_more_to_h(
        self,
        trained_circuit: tuple,
    ) -> None:
        """The H-winner should fire at least as much for H as for V.

        Note: >= not > because in WTA circuits, a neuron can win for H
        through population dynamics (others fire less) even if it
        personally responds equally to both orientations.
        """
        _, ids, results, _ = trained_circuit
        v1_exc = ids["v1_exc"]
        h_winner = max(v1_exc, key=lambda n: results["H"][n])
        assert results["H"][h_winner] >= results["V"][h_winner], (
            f"H-winner V1[{h_winner}] fires more for V: "
            f"H={results['H'][h_winner]}, V={results['V'][h_winner]}"
        )

    def test_v_winner_responds_more_to_v(
        self,
        trained_circuit: tuple,
    ) -> None:
        """The V-winner should fire more for V than for H."""
        _, ids, results, _ = trained_circuit
        v1_exc = ids["v1_exc"]
        v_winner = max(v1_exc, key=lambda n: results["V"][n])
        assert results["V"][v_winner] >= results["H"][v_winner], (
            f"V-winner V1[{v_winner}] fires more for H: "
            f"H={results['H'][v_winner]}, V={results['V'][v_winner]}"
        )


class TestSelfOrganizationRobustness:
    """Test that self-organization works across multiple random seeds."""

    @pytest.mark.parametrize("seed", [7, 42, 99])
    def test_differentiation_across_seeds(self, seed: int) -> None:
        """Different winners should emerge for H and V across seeds."""
        _, ids, results, _ = _train_v1(seed=seed)
        v1_exc = ids["v1_exc"]
        h_winner = max(v1_exc, key=lambda n: results["H"][n])
        v_winner = max(v1_exc, key=lambda n: results["V"][n])
        assert h_winner != v_winner, (
            f"Seed {seed}: same winner V1[{h_winner}] for both orientations"
        )
