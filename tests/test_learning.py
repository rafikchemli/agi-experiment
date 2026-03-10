"""Experiments 3-4: Lateral inhibition, WTA, and homeostatic plasticity.

Experiment 3 tests: Canonical cortical microcircuit (excitatory neurons
connected through a shared inhibitory pool) produces competitive dynamics.
Assumptions tested: A6 (WTA via shared inhibitory pool)

Experiment 4 tests: Homeostatic synaptic scaling prevents STDP-driven
runaway excitation and rescues silent neurons.
Assumptions tested: A7 (homeostatic scaling is sufficient for stability)
"""

import pytest

from brain_sim.learning import apply_homeostatic_scaling
from brain_sim.network import Network, build_wta_circuit
from brain_sim.neuron import IzhikevichNeuron, NeuronType
from brain_sim.synapse import STDPParams, Synapse, SynapseType

# --- WTA circuit parameters tuned empirically ---
WTA_PARAMS = {
    "n_excitatory": 5,
    "n_inhibitory": 2,
    "w_exc_to_inh": 8.0,
    "w_inh_to_exc": 20.0,
    "plasticity": False,
    "tau_syn": 8.0,
}
INH_TONIC = 4.0  # tonic current to keep inhibitory neurons near threshold
SIM_DURATION = 500  # ms


def _run_wta(drives: dict[int, float]) -> dict[int, int]:
    """Run a WTA circuit with given excitatory drives and return spike counts.

    Args:
        drives: External current per excitatory neuron ID (0..4).

    Returns:
        Spike counts for excitatory neurons only (IDs 0..4).
    """
    net = build_wta_circuit(**WTA_PARAMS)
    currents = dict(drives)
    for i_id in range(
        WTA_PARAMS["n_excitatory"], WTA_PARAMS["n_excitatory"] + WTA_PARAMS["n_inhibitory"]
    ):
        currents[i_id] = INH_TONIC
    net.simulate(SIM_DURATION, external_currents=currents)
    counts = net.get_spike_counts()
    return {k: v for k, v in counts.items() if k < WTA_PARAMS["n_excitatory"]}


class TestNetworkBasics:
    """Verify basic network assembly and simulation mechanics."""

    def test_add_neurons(self) -> None:
        """Network should track added neurons."""
        net = Network()
        net.add_neuron(0, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        net.add_neuron(1, IzhikevichNeuron.from_type(NeuronType.FAST_SPIKING))
        assert len(net.neurons) == 2

    def test_duplicate_neuron_raises(self) -> None:
        """Adding a neuron with an existing ID should raise ValueError."""
        net = Network()
        net.add_neuron(0, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        with pytest.raises(ValueError, match="already exists"):
            net.add_neuron(0, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))

    def test_connect_invalid_neuron_raises(self) -> None:
        """Connecting to a non-existent neuron should raise ValueError."""
        net = Network()
        net.add_neuron(0, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        with pytest.raises(ValueError, match="not in network"):
            net.connect(0, 99, Synapse(weight=1.0, synapse_type=SynapseType.EXCITATORY))

    def test_isolated_neuron_fires_with_current(self) -> None:
        """A single neuron in the network should fire with sufficient current."""
        net = Network()
        net.add_neuron(0, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        spikes = net.simulate(500, external_currents={0: 10.0})
        assert len(spikes[0]) > 0, "Isolated neuron did not fire with I=10"

    def test_spike_propagates_through_synapse(self) -> None:
        """A spike in pre should deliver current to post via synapse."""
        net = Network(plasticity=False, tau_syn=5.0)
        net.add_neuron(0, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        net.add_neuron(1, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        net.connect(
            0,
            1,
            Synapse(weight=10.0, synapse_type=SynapseType.EXCITATORY, delay=1),
        )
        # Drive neuron 0 strongly, neuron 1 at subthreshold
        spikes = net.simulate(500, external_currents={0: 15.0, 1: 3.0})
        # Neuron 0 should fire, and its spikes should help neuron 1 fire too
        assert len(spikes[0]) > 0, "Pre neuron did not fire"
        assert len(spikes[1]) > 0, "Post neuron did not fire — synaptic current not propagating"

    def test_reset_clears_all_state(self) -> None:
        """Reset should clear spike logs, PSCs, and neuron states."""
        net = Network()
        net.add_neuron(0, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        net.simulate(100, external_currents={0: 10.0})
        assert len(net._spike_log[0]) > 0
        net.reset()
        assert len(net._spike_log[0]) == 0
        assert net._psc[0] == 0.0
        assert net._timestep == 0


class TestPSCDecay:
    """Verify post-synaptic current integration and decay."""

    def test_psc_decays_between_spikes(self) -> None:
        """PSC should decay exponentially when no new spikes arrive."""
        import math

        net = Network(plasticity=False, tau_syn=5.0)
        net.add_neuron(0, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        net.add_neuron(1, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        net.connect(
            0,
            1,
            Synapse(weight=10.0, synapse_type=SynapseType.EXCITATORY, delay=1),
        )
        # Give neuron 0 a strong kick to fire once quickly
        net.step(external_currents={0: 100.0})
        # The spike from 0 will arrive at 1 after delay=1
        net.step(external_currents={0: 0.0})
        # Now PSC for neuron 1 should be ~10.0
        psc_after_delivery = net._psc[1]
        assert psc_after_delivery > 0, "PSC should be positive after spike delivery"

        # After one more step, PSC should decay
        net.step(external_currents={0: 0.0})
        expected_decay = psc_after_delivery * math.exp(-1.0 / 5.0)
        assert abs(net._psc[1] - expected_decay) < 0.5, (
            f"PSC decay incorrect: expected ~{expected_decay:.2f}, got {net._psc[1]:.2f}"
        )


class TestWinnerTakeAll:
    """Experiment 3: Lateral inhibition produces WTA dynamics.

    A WTA circuit with 5 excitatory (RS) neurons and 2 inhibitory (FS)
    neurons. Each E connects to the shared I pool, and I connects back
    to all E neurons. The strongest-driven E neuron should dominate.
    """

    def test_strongest_input_wins(self) -> None:
        """The neuron with the strongest external drive should fire the most."""
        drives = {0: 5.0, 1: 7.0, 2: 15.0, 3: 6.0, 4: 4.0}
        counts = _run_wta(drives)
        winner = max(counts, key=lambda k: counts[k])
        assert winner == 2, (
            f"Expected neuron 2 (I=15) to win, but neuron {winner} won. Counts: {counts}"
        )

    def test_winner_fires_more_than_any_loser(self) -> None:
        """The winner should fire at least 50% more than the strongest loser."""
        drives = {0: 5.0, 1: 7.0, 2: 15.0, 3: 6.0, 4: 4.0}
        counts = _run_wta(drives)
        winner_count = counts[2]
        loser_counts = [v for k, v in counts.items() if k != 2]
        max_loser = max(loser_counts)
        assert winner_count > max_loser * 1.5, (
            f"Winner ({winner_count}) should fire >1.5x strongest loser ({max_loser}). "
            f"Counts: {counts}"
        )

    def test_changing_winner(self) -> None:
        """Switching which neuron gets the strongest drive should change the winner."""
        # E0 is strongest
        drives_0 = {0: 15.0, 1: 7.0, 2: 5.0, 3: 6.0, 4: 4.0}
        counts_0 = _run_wta(drives_0)
        winner_0 = max(counts_0, key=lambda k: counts_0[k])

        # E4 is strongest
        drives_4 = {0: 5.0, 1: 7.0, 2: 6.0, 3: 4.0, 4: 15.0}
        counts_4 = _run_wta(drives_4)
        winner_4 = max(counts_4, key=lambda k: counts_4[k])

        assert winner_0 == 0, f"Expected E0 to win, got {winner_0}. Counts: {counts_0}"
        assert winner_4 == 4, f"Expected E4 to win, got {winner_4}. Counts: {counts_4}"

    def test_inhibition_suppresses_losers(self) -> None:
        """Neurons in the WTA circuit should fire less than in isolation.

        Feedback inhibition from the shared pool should suppress weaker neurons
        compared to how they'd fire without any inhibition.
        """
        drives = {0: 5.0, 1: 7.0, 2: 15.0, 3: 6.0, 4: 4.0}

        # Run without inhibition (isolated neurons)
        alone = Network(plasticity=False)
        for i in range(5):
            alone.add_neuron(i, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        alone.simulate(SIM_DURATION, external_currents=drives)
        alone_counts = alone.get_spike_counts()

        # Run with WTA
        wta_counts = _run_wta(drives)

        # At least some losers should be suppressed
        suppressed = 0
        for nid in range(5):
            if nid == 2:
                continue  # skip winner
            if wta_counts[nid] < alone_counts[nid]:
                suppressed += 1

        assert suppressed >= 2, (
            f"Expected at least 2 non-winner neurons to be suppressed. "
            f"Alone: {alone_counts}, WTA: {wta_counts}"
        )

    def test_equal_drives_no_clear_winner(self) -> None:
        """With equal drives, all excitatory neurons should fire similarly.

        No neuron has an advantage, so the inhibition affects all equally.
        """
        drives = {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}
        counts = _run_wta(drives)
        values = list(counts.values())
        avg = sum(values) / len(values)
        # All should be within 30% of average
        for nid, count in counts.items():
            assert abs(count - avg) <= avg * 0.3 + 1, (
                f"Neuron {nid} fired {count} times, expected ~{avg:.0f}. "
                f"Equal drives should give equal firing. Counts: {counts}"
            )

    def test_close_competition(self) -> None:
        """Two neurons with similar strong drives should both fire significantly.

        Soft WTA: the winner dominates but doesn't completely silence a close
        competitor. This is biologically realistic — partial suppression.
        """
        drives = {0: 5.0, 1: 14.0, 2: 15.0, 3: 5.0, 4: 5.0}
        counts = _run_wta(drives)
        # E2 should win (or tie) but E1 should also fire substantially
        assert counts[2] >= counts[1], f"E2 (I=15) should fire >= E1 (I=14). Counts: {counts}"
        assert counts[1] > 5, (
            f"E1 (I=14) should still fire substantially in soft WTA, "
            f"but only fired {counts[1]}. Counts: {counts}"
        )


class TestBuildWtaCircuit:
    """Verify the WTA circuit builder produces correct topology."""

    def test_correct_neuron_count(self) -> None:
        """Should create n_excitatory + n_inhibitory neurons."""
        net = build_wta_circuit(n_excitatory=5, n_inhibitory=2)
        assert len(net.neurons) == 7

    def test_correct_connection_count(self) -> None:
        """Each E connects to each I, and each I connects to each E.

        Total connections = n_exc * n_inh + n_inh * n_exc = 2 * n_exc * n_inh.
        """
        net = build_wta_circuit(n_excitatory=5, n_inhibitory=2)
        expected = 2 * 5 * 2  # E->I + I->E
        assert len(net._connections) == expected, (
            f"Expected {expected} connections, got {len(net._connections)}"
        )

    def test_excitatory_connections_positive(self) -> None:
        """E->I synapses should have positive weights."""
        net = build_wta_circuit(n_excitatory=3, n_inhibitory=1)
        for conn in net._connections:
            if conn.pre_id < 3:  # excitatory pre
                assert conn.synapse.weight > 0, (
                    f"E->I synapse ({conn.pre_id}->{conn.post_id}) "
                    f"has non-positive weight {conn.synapse.weight}"
                )

    def test_inhibitory_connections_negative(self) -> None:
        """I->E synapses should have negative weights (Dale's Law)."""
        net = build_wta_circuit(n_excitatory=3, n_inhibitory=1)
        for conn in net._connections:
            if conn.pre_id >= 3:  # inhibitory pre
                assert conn.synapse.weight < 0, (
                    f"I->E synapse ({conn.pre_id}->{conn.post_id}) "
                    f"has non-negative weight {conn.synapse.weight}"
                )


# --- Experiment 4 helpers ---

# Custom STDP with higher w_max for circuit-level weights
_CIRCUIT_STDP = STDPParams(
    a_plus=0.01,
    a_minus=0.012,
    tau_plus=20.0,
    tau_minus=20.0,
    w_max=15.0,
    w_min=0.0,
)


def _build_input_output_net(
    n_inputs: int = 5,
    initial_weight: float = 8.0,
    plasticity: bool = True,
) -> Network:
    """Build a simple input -> output network for homeostasis testing.

    Args:
        n_inputs: Number of input neurons.
        initial_weight: Initial synapse weight from each input to output.
        plasticity: Whether STDP is active.

    Returns:
        Network with input neurons (IDs 0..n_inputs-1) and one output
        neuron (ID n_inputs).
    """
    net = Network(plasticity=plasticity, tau_syn=5.0)
    for i in range(n_inputs):
        net.add_neuron(i, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
    out_id = n_inputs
    net.add_neuron(out_id, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
    for i in range(n_inputs):
        net.connect(
            i,
            out_id,
            Synapse(
                weight=initial_weight,
                synapse_type=SynapseType.EXCITATORY,
                stdp_params=_CIRCUIT_STDP,
            ),
        )
    return net


class TestHomeostaticScaling:
    """Experiment 4: Homeostatic plasticity stabilizes firing rates.

    Without homeostasis, STDP creates a positive feedback loop:
    more firing -> more LTP -> higher weights -> more firing.
    Homeostatic scaling counteracts this by adjusting weights
    to maintain a target firing rate (Turrigiano et al., 1998).
    """

    def test_scaling_reduces_weights_when_rate_too_high(self) -> None:
        """If a neuron fires above target, homeostasis should scale down."""
        net = _build_input_output_net(n_inputs=5, initial_weight=8.0, plasticity=False)
        drives = {i: 10.0 for i in range(5)}
        net.simulate(1000, external_currents=drives)

        out_id = 5
        rate = net.recent_spike_count(out_id, 1000)
        assert rate > 15, f"Output should fire > 15Hz, got {rate}"

        weights_before = [c.synapse.weight for c in net.incoming_connections(out_id)]
        factors = apply_homeostatic_scaling(net, {out_id: 15.0}, window_ms=1000, eta=0.05)
        weights_after = [c.synapse.weight for c in net.incoming_connections(out_id)]

        assert factors[out_id] < 1.0, "Scale factor should be < 1 when rate > target"
        assert all(a < b for a, b in zip(weights_after, weights_before, strict=True)), (
            "All weights should decrease"
        )

    def test_scaling_increases_weights_when_rate_too_low(self) -> None:
        """If a neuron fires below target, homeostasis should scale up."""
        net = _build_input_output_net(n_inputs=5, initial_weight=1.0, plasticity=False)
        drives = {i: 10.0 for i in range(5)}
        net.simulate(1000, external_currents=drives)

        out_id = 5
        rate = net.recent_spike_count(out_id, 1000)
        assert rate < 15, f"Output should fire < 15Hz with weak weights, got {rate}"

        weights_before = [c.synapse.weight for c in net.incoming_connections(out_id)]
        factors = apply_homeostatic_scaling(net, {out_id: 15.0}, window_ms=1000, eta=0.05)
        weights_after = [c.synapse.weight for c in net.incoming_connections(out_id)]

        assert factors[out_id] > 1.0, "Scale factor should be > 1 when rate < target"
        assert all(a > b for a, b in zip(weights_after, weights_before, strict=True)), (
            "All weights should increase"
        )

    def test_scaling_ignores_inhibitory_synapses(self) -> None:
        """Homeostasis should only scale excitatory incoming synapses."""
        net = Network(plasticity=False, tau_syn=5.0)
        net.add_neuron(0, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))
        net.add_neuron(1, IzhikevichNeuron.from_type(NeuronType.FAST_SPIKING))
        net.add_neuron(2, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))

        # Excitatory input
        net.connect(0, 2, Synapse(weight=5.0, synapse_type=SynapseType.EXCITATORY))
        # Inhibitory input
        inh_syn = Synapse(weight=5.0, synapse_type=SynapseType.INHIBITORY)
        net.connect(1, 2, inh_syn)

        inh_weight_before = inh_syn.weight
        net.simulate(1000, external_currents={0: 10.0, 1: 10.0})
        apply_homeostatic_scaling(net, {2: 15.0}, window_ms=1000, eta=0.1)

        assert inh_syn.weight == inh_weight_before, (
            f"Inhibitory weight changed: {inh_weight_before} -> {inh_syn.weight}"
        )

    def test_stdp_runaway_without_homeostasis(self) -> None:
        """Without homeostasis, STDP should cause monotonic weight growth.

        This demonstrates the problem that homeostasis solves.
        """
        net = _build_input_output_net(n_inputs=5, initial_weight=8.0, plasticity=True)
        out_id = 5
        drives = {i: 10.0 for i in range(5)}

        # Run 5 epochs, record average weight after each
        avg_weights: list[float] = []
        for _ in range(5):
            net.simulate(1000, external_currents=drives)
            ws = [c.synapse.weight for c in net.incoming_connections(out_id)]
            avg_weights.append(sum(ws) / len(ws))

        # Weights should be monotonically increasing (STDP positive feedback)
        for i in range(1, len(avg_weights)):
            assert avg_weights[i] >= avg_weights[i - 1] - 0.01, (
                f"Expected monotonic weight growth without homeostasis. "
                f"Epoch {i - 1}: {avg_weights[i - 1]:.3f}, Epoch {i}: {avg_weights[i]:.3f}"
            )
        assert avg_weights[-1] > avg_weights[0], (
            f"Weights should grow over time without homeostasis: "
            f"start={avg_weights[0]:.3f}, end={avg_weights[-1]:.3f}"
        )

    def test_homeostasis_prevents_runaway(self) -> None:
        """With homeostasis, STDP weight growth should be bounded.

        Homeostasis counteracts STDP's tendency to increase weights
        when the neuron fires above its target rate.
        """
        net = _build_input_output_net(n_inputs=5, initial_weight=8.0, plasticity=True)
        out_id = 5
        drives = {i: 10.0 for i in range(5)}

        # Run 10 epochs with homeostasis applied after each
        avg_weights: list[float] = []
        for _ in range(10):
            net.simulate(1000, external_currents=drives)
            apply_homeostatic_scaling(net, {out_id: 15.0}, window_ms=1000, eta=0.05)
            ws = [c.synapse.weight for c in net.incoming_connections(out_id)]
            avg_weights.append(sum(ws) / len(ws))

        # Weight growth should be bounded — final weight should be less
        # than it would be without homeostasis (which exceeded 11.0)
        assert avg_weights[-1] < 10.0, (
            f"Homeostasis should prevent runaway weight growth. "
            f"Final avg weight: {avg_weights[-1]:.3f}"
        )

    def test_homeostasis_rescues_silent_neuron(self) -> None:
        """A neuron with too-weak input should be gradually rescued.

        Homeostatic scaling boosts weights when the neuron fires below target,
        eventually bringing it above threshold.
        """
        net = _build_input_output_net(n_inputs=5, initial_weight=1.0, plasticity=False)
        out_id = 5
        drives = {i: 10.0 for i in range(5)}

        # Initially the output should be silent or near-silent
        net.simulate(1000, external_currents=drives)
        initial_rate = net.recent_spike_count(out_id, 1000)

        # Apply homeostasis for several epochs
        for _ in range(15):
            apply_homeostatic_scaling(net, {out_id: 15.0}, window_ms=1000, eta=0.1)
            net.simulate(1000, external_currents=drives)

        final_rate = net.recent_spike_count(out_id, 1000)
        assert final_rate > initial_rate, (
            f"Homeostasis should rescue silent neuron. "
            f"Initial rate: {initial_rate}, Final rate: {final_rate}"
        )
        assert final_rate > 5, (
            f"After 15 epochs of homeostasis, neuron should fire > 5Hz. Got {final_rate}Hz"
        )
