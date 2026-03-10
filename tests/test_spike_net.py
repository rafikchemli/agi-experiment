"""Tests for vectorized SpikeNetwork.

Verifies that the vectorized implementation produces the same biological
behaviors as the per-object Network:
1. Izhikevich dynamics (firing rates, adaptation)
2. WTA competition via lateral inhibition
3. STDP plasticity (LTP, LTD, timing dependence)
4. Self-organization with STDP + WTA + homeostasis
"""

import numpy as np
import pytest

from brain_sim.neuron import NeuronType
from brain_sim.spike_net import (
    Population,
    Projection,
    SpikeNetwork,
    build_v1_circuit_vectorized,
)
from brain_sim.synapse import STDPParams, SynapseType


class TestPopulation:
    """Test vectorized neuron population."""

    def test_resting_state(self) -> None:
        pop = Population(10, NeuronType.REGULAR_SPIKING, "test")
        assert pop.v.shape == (10,)
        assert np.all(pop.v == -65.0)
        assert not pop.fired.any()

    def test_fires_with_current(self) -> None:
        pop = Population(3, NeuronType.REGULAR_SPIKING, "test")
        # Drive with high current for a while
        for _ in range(50):
            pop.step(np.array([10.0, 0.0, 10.0]))
        assert pop.spike_counts[0] > 0
        assert pop.spike_counts[1] == 0
        assert pop.spike_counts[2] > 0

    def test_no_spontaneous_firing(self) -> None:
        pop = Population(5, NeuronType.REGULAR_SPIKING, "test")
        for _ in range(100):
            pop.step(np.zeros(5))
        assert pop.spike_counts.sum() == 0

    def test_fs_faster_than_rs(self) -> None:
        rs = Population(1, NeuronType.REGULAR_SPIKING, "rs")
        fs = Population(1, NeuronType.FAST_SPIKING, "fs")
        for _ in range(500):
            rs.step(np.array([10.0]))
            fs.step(np.array([10.0]))
        assert fs.spike_counts[0] > rs.spike_counts[0]

    def test_recent_spike_count(self) -> None:
        pop = Population(1, NeuronType.REGULAR_SPIKING, "test")
        for _ in range(200):
            pop.step(np.array([10.0]))
        total = pop.spike_counts[0]
        recent = pop.recent_spike_count(100)[0]
        assert recent > 0
        assert recent <= total

    def test_reset(self) -> None:
        pop = Population(2, NeuronType.REGULAR_SPIKING, "test")
        for _ in range(100):
            pop.step(np.array([10.0, 10.0]))
        pop.reset()
        assert np.all(pop.v == -65.0)
        assert pop.spike_counts.sum() == 0
        assert not pop.fired.any()


class TestProjection:
    """Test vectorized synaptic projections."""

    def test_weight_shape(self) -> None:
        pre = Population(4, NeuronType.REGULAR_SPIKING, "pre")
        post = Population(3, NeuronType.REGULAR_SPIKING, "post")
        w = np.ones((4, 3))
        proj = Projection(pre, post, w, SynapseType.EXCITATORY)
        assert proj.weights.shape == (4, 3)

    def test_wrong_shape_raises(self) -> None:
        pre = Population(4, NeuronType.REGULAR_SPIKING, "pre")
        post = Population(3, NeuronType.REGULAR_SPIKING, "post")
        with pytest.raises(ValueError, match="does not match"):
            Projection(pre, post, np.ones((3, 3)), SynapseType.EXCITATORY)

    def test_dales_law_excitatory(self) -> None:
        pre = Population(2, NeuronType.REGULAR_SPIKING, "pre")
        post = Population(2, NeuronType.REGULAR_SPIKING, "post")
        w = np.array([[1.0, -0.5], [0.5, 1.0]])
        proj = Projection(pre, post, w, SynapseType.EXCITATORY)
        assert np.all(proj.weights >= 0)

    def test_dales_law_inhibitory(self) -> None:
        pre = Population(2, NeuronType.FAST_SPIKING, "pre")
        post = Population(2, NeuronType.REGULAR_SPIKING, "post")
        w = np.array([[5.0, 3.0], [4.0, 6.0]])
        proj = Projection(pre, post, w, SynapseType.INHIBITORY)
        assert np.all(proj.weights <= 0)

    def test_current_delivery(self) -> None:
        pre = Population(2, NeuronType.REGULAR_SPIKING, "pre")
        post = Population(2, NeuronType.REGULAR_SPIKING, "post")
        stdp = STDPParams(w_max=10.0, w_min=0.0)
        w = np.array([[1.0, 2.0], [3.0, 4.0]])
        proj = Projection(pre, post, w, SynapseType.EXCITATORY, stdp_params=stdp)
        # Simulate pre[0] firing
        pre.fired = np.array([True, False])
        proj.deliver_and_ltd(pre.fired, apply_stdp=False)
        # post.psc should be W.T @ [1, 0] = [1.0, 2.0]
        np.testing.assert_allclose(post.psc, [1.0, 2.0])


class TestSpikeNetwork:
    """Test network assembly and simulation."""

    def test_add_populations(self) -> None:
        net = SpikeNetwork()
        net.add_population(Population(5, NeuronType.REGULAR_SPIKING, "exc"))
        net.add_population(Population(2, NeuronType.FAST_SPIKING, "inh"))
        assert len(net.populations) == 2

    def test_duplicate_population_raises(self) -> None:
        net = SpikeNetwork()
        net.add_population(Population(5, NeuronType.REGULAR_SPIKING, "test"))
        with pytest.raises(ValueError, match="already exists"):
            net.add_population(Population(3, NeuronType.REGULAR_SPIKING, "test"))

    def test_spike_propagates(self) -> None:
        """Pre neuron driven to fire → current delivered to post."""
        pre = Population(1, NeuronType.REGULAR_SPIKING, "pre")
        post = Population(1, NeuronType.REGULAR_SPIKING, "post")
        stdp = STDPParams(w_max=15.0, w_min=0.0)
        w = np.array([[8.0]])
        proj = Projection(pre, post, w, SynapseType.EXCITATORY, plastic=False, stdp_params=stdp)
        net = SpikeNetwork(tau_syn=5.0, plasticity=False)
        net.add_population(pre)
        net.add_population(post)
        net.add_projection(proj)
        # Drive pre strongly
        for _ in range(200):
            net.step({"pre": np.array([15.0])})
        # Post should have fired from synaptic current
        assert post.spike_counts[0] > 0

    def test_wta_competition(self) -> None:
        """Strongest-driven neuron should fire the most in a WTA circuit."""
        n_exc = 3
        n_inh = 1
        exc = Population(n_exc, NeuronType.REGULAR_SPIKING, "exc")
        inh = Population(n_inh, NeuronType.FAST_SPIKING, "inh")

        w_ei = np.full((n_exc, n_inh), 8.0)
        w_ie = np.full((n_inh, n_exc), 20.0)

        net = SpikeNetwork(tau_syn=8.0, plasticity=False)
        net.add_population(exc)
        net.add_population(inh)
        net.add_projection(Projection(exc, inh, w_ei, SynapseType.EXCITATORY, plastic=False))
        net.add_projection(Projection(inh, exc, w_ie, SynapseType.INHIBITORY, plastic=False))

        # Drive E[1] strongest
        drives = np.array([5.0, 15.0, 5.0])
        inh_tonic = np.array([4.0])
        for _ in range(500):
            net.step({"exc": drives, "inh": inh_tonic})

        winner = int(np.argmax(exc.spike_counts))
        assert winner == 1

    def test_simulate_returns_counts(self) -> None:
        pop = Population(2, NeuronType.REGULAR_SPIKING, "test")
        net = SpikeNetwork()
        net.add_population(pop)
        result = net.simulate(100, {"test": np.array([10.0, 0.0])})
        assert result["test"][0] > 0
        assert result["test"][1] == 0

    def test_reset(self) -> None:
        pop = Population(2, NeuronType.REGULAR_SPIKING, "test")
        net = SpikeNetwork()
        net.add_population(pop)
        net.simulate(100, {"test": np.array([10.0, 10.0])})
        net.reset()
        assert pop.spike_counts.sum() == 0
        assert np.all(pop.v == -65.0)


class TestSTDP:
    """Test that STDP produces correct plasticity in the vectorized version."""

    def test_pre_before_post_strengthens(self) -> None:
        """Causal timing (pre then post) should strengthen the synapse."""
        pre = Population(1, NeuronType.REGULAR_SPIKING, "pre")
        post = Population(1, NeuronType.REGULAR_SPIKING, "post")
        stdp = STDPParams(
            a_plus=0.01,
            a_minus=0.012,
            tau_plus=20.0,
            tau_minus=20.0,
            w_max=1.0,
            w_min=0.0,
        )
        w_init = 0.5
        w = np.array([[w_init]])
        proj = Projection(pre, post, w, SynapseType.EXCITATORY, plastic=True, stdp_params=stdp)
        net = SpikeNetwork(plasticity=True)
        net.add_population(pre)
        net.add_population(post)
        net.add_projection(proj)

        # Force pre spike at t=10, post spike at t=15 (causal, dt=+5ms)
        for t in range(50):
            pre_curr = np.array([20.0]) if t == 10 else np.zeros(1)
            post_curr = np.array([20.0]) if t == 15 else np.zeros(1)
            net.step({"pre": pre_curr, "post": post_curr})

        assert proj.weights[0, 0] > w_init

    def test_plastic_flag_prevents_changes(self) -> None:
        """Non-plastic projections should not have weight changes."""
        pre = Population(1, NeuronType.REGULAR_SPIKING, "pre")
        post = Population(1, NeuronType.REGULAR_SPIKING, "post")
        w_init = 0.5
        w = np.array([[w_init]])
        proj = Projection(pre, post, w, SynapseType.EXCITATORY, plastic=False)
        net = SpikeNetwork(plasticity=True)
        net.add_population(pre)
        net.add_population(post)
        net.add_projection(proj)

        for _ in range(100):
            net.step({"pre": np.array([15.0]), "post": np.array([15.0])})

        assert proj.weights[0, 0] == w_init


class TestBuildV1Vectorized:
    """Test the vectorized V1 circuit builder."""

    def test_population_sizes(self) -> None:
        net, pops, _ = build_v1_circuit_vectorized(
            grid_size=4,
            n_v1_excitatory=3,
            n_v1_inhibitory=1,
        )
        assert pops["retina"].n == 16
        assert pops["v1_exc"].n == 3
        assert pops["v1_inh"].n == 1
        assert len(net.projections) == 3

    def test_input_projection_is_plastic(self) -> None:
        net, _, _ = build_v1_circuit_vectorized(grid_size=4)
        # First projection is retina -> v1_exc
        assert net.projections[0].plastic

    def test_lateral_projections_are_fixed(self) -> None:
        net, _, _ = build_v1_circuit_vectorized(grid_size=4)
        # Projections 1 and 2 are V1_exc->V1_inh and V1_inh->V1_exc
        assert not net.projections[1].plastic
        assert not net.projections[2].plastic

    def test_self_organization(self) -> None:
        """The vectorized network should self-organize like the original."""
        from brain_sim.encoding import (
            generate_horizontal_bar,
            generate_vertical_bar,
        )

        rng = np.random.default_rng(42)
        stdp = STDPParams(
            a_plus=0.015,
            a_minus=0.016,
            tau_plus=20.0,
            tau_minus=20.0,
            w_max=10.0,
            w_min=0.0,
        )
        grid = 4
        net, pops, inh_tonic = build_v1_circuit_vectorized(
            grid_size=grid,
            n_v1_excitatory=4,
            n_v1_inhibitory=2,
            w_input_to_v1=4.0,
            w_v1_to_inh=6.0,
            w_inh_to_v1=18.0,
            inh_tonic=3.0,
            tau_syn=5.0,
            stdp_params=stdp,
            rng=rng,
        )
        h_bar = generate_horizontal_bar(grid, row=grid // 2, width=1)
        v_bar = generate_vertical_bar(grid, col=grid // 2, width=1)
        patterns = [h_bar, v_bar]

        # Train
        for _epoch in range(100):
            order = list(range(len(patterns)))
            rng.shuffle(order)
            for idx in order:
                retina_curr = patterns[idx].flatten() * 15.0
                inh_curr = np.full(pops["v1_inh"].n, inh_tonic)
                net.simulate(150, {"retina": retina_curr, "v1_inh": inh_curr})

            # Homeostasis
            proj_in = net.projections[0]
            recent = pops["v1_exc"].recent_spike_count(300)
            for j in range(pops["v1_exc"].n):
                actual_hz = recent[j] / 0.3
                factor = 1.0 + 0.08 * (8.0 - actual_hz) / 8.0 if actual_hz > 0 else 1.08
                factor = max(0.8, min(1.2, factor))
                proj_in.weights[:, j] *= factor
            np.clip(
                proj_in.weights,
                proj_in.stdp.w_min,
                proj_in.stdp.w_max,
                out=proj_in.weights,
            )

        # Test: different winners for H and V
        results: dict[str, np.ndarray] = {}
        for label, img in [("H", h_bar), ("V", v_bar)]:
            net.reset()
            retina_curr = img.flatten() * 15.0
            inh_curr = np.full(pops["v1_inh"].n, inh_tonic)
            net.simulate(500, {"retina": retina_curr, "v1_inh": inh_curr})
            results[label] = pops["v1_exc"].spike_counts.copy()

        h_winner = int(np.argmax(results["H"]))
        v_winner = int(np.argmax(results["V"]))
        assert h_winner != v_winner, f"Same winner for both: H={results['H']}, V={results['V']}"
