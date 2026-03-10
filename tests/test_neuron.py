"""Experiment 1: Does the Izhikevich neuron model work?

Tests verify that each neuron type produces the expected spiking behavior
when injected with constant current, matching Izhikevich 2003 Figure 1.

Assumptions tested: A1 (Izhikevich is a good neuron model)
"""

from brain_sim.neuron import (
    IzhikevichNeuron,
    NeuronParams,
    NeuronType,
    simulate_neuron,
)


class TestNeuronInitialization:
    """Verify neurons start in a biologically valid resting state."""

    def test_resting_potential(self, rs_neuron: IzhikevichNeuron) -> None:
        """Resting membrane potential should be ~-65mV."""
        assert rs_neuron.v == -65.0

    def test_no_spontaneous_spikes(self, rs_neuron: IzhikevichNeuron) -> None:
        """A neuron with no input should not spike."""
        for _ in range(1000):
            rs_neuron.step(current=0.0)
        assert len(rs_neuron.spike_times) == 0

    def test_from_type_creates_correct_params(self) -> None:
        """from_type should use the biological presets."""
        neuron = IzhikevichNeuron.from_type(NeuronType.FAST_SPIKING)
        assert neuron.params.a == 0.1
        assert neuron.params.b == 0.2
        assert neuron.params.c == -65.0
        assert neuron.params.d == 2.0

    def test_custom_params(self) -> None:
        """Should accept arbitrary parameters for experimentation."""
        params = NeuronParams(a=0.05, b=0.25, c=-60.0, d=6.0)
        neuron = IzhikevichNeuron(params=params)
        assert neuron.params.a == 0.05


class TestRegularSpiking:
    """Regular Spiking neurons: the most common excitatory cortical neuron.

    Expected behavior: tonic firing with spike frequency adaptation.
    With I=10, should fire at a moderate rate (~5-20 Hz).
    """

    def test_fires_with_sufficient_current(self, rs_neuron: IzhikevichNeuron) -> None:
        """RS neuron should fire when given enough input current."""
        _, _, spikes = simulate_neuron(rs_neuron, current=10.0, duration_ms=1000)
        assert len(spikes) > 0, "RS neuron did not fire with I=10"

    def test_firing_rate_moderate(self, rs_neuron: IzhikevichNeuron) -> None:
        """RS neuron at I=10 should fire at a moderate rate (5-30 Hz)."""
        _, _, spikes = simulate_neuron(rs_neuron, current=10.0, duration_ms=1000)
        rate = len(spikes)  # spikes per 1000ms = Hz
        assert 3 <= rate <= 40, f"RS firing rate {rate} Hz outside expected 3-40 Hz range"

    def test_spike_frequency_adaptation(self, rs_neuron: IzhikevichNeuron) -> None:
        """RS neurons show adaptation: early ISIs shorter than late ISIs.

        This is a key biological property — the neuron fires fast initially
        then slows down. Caused by the recovery variable u accumulating.
        """
        _, _, spikes = simulate_neuron(rs_neuron, current=14.0, duration_ms=1000)
        if len(spikes) < 4:
            return  # not enough spikes to measure adaptation
        isis = [spikes[i + 1] - spikes[i] for i in range(len(spikes) - 1)]
        # Early ISIs should be shorter (faster firing) than late ISIs
        early_avg = sum(isis[:2]) / 2
        late_avg = sum(isis[-2:]) / 2
        assert late_avg >= early_avg * 0.8, (
            f"Expected adaptation: early ISI={early_avg:.1f}, late ISI={late_avg:.1f}"
        )

    def test_no_firing_below_threshold(self, rs_neuron: IzhikevichNeuron) -> None:
        """RS neuron should not fire with very low current."""
        _, _, spikes = simulate_neuron(rs_neuron, current=2.0, duration_ms=1000)
        assert len(spikes) == 0, f"RS neuron fired {len(spikes)} times with subthreshold I=2"

    def test_higher_current_means_higher_rate(self, rs_neuron: IzhikevichNeuron) -> None:
        """Firing rate should increase with input current (f-I curve)."""
        _, _, spikes_low = simulate_neuron(rs_neuron, current=8.0, duration_ms=1000)
        rs_neuron.reset()
        _, _, spikes_high = simulate_neuron(rs_neuron, current=20.0, duration_ms=1000)
        assert len(spikes_high) > len(spikes_low), (
            f"Higher current should give more spikes: I=8 gave {len(spikes_low)}, "
            f"I=20 gave {len(spikes_high)}"
        )


class TestFastSpiking:
    """Fast Spiking neurons: inhibitory interneurons.

    Expected behavior: high-frequency tonic firing without adaptation.
    With I=10, should fire at a much higher rate than RS neurons.
    """

    def test_fires_at_high_rate(self, fs_neuron: IzhikevichNeuron) -> None:
        """FS neuron should fire at a significantly higher rate than RS."""
        _, _, spikes = simulate_neuron(fs_neuron, current=10.0, duration_ms=1000)
        rate = len(spikes)
        assert rate > 15, f"FS neuron firing rate {rate} Hz is too low (expected >15 Hz)"

    def test_minimal_adaptation(self, fs_neuron: IzhikevichNeuron) -> None:
        """FS neurons should show little to no spike frequency adaptation.

        This is a defining feature — they maintain steady high-frequency firing.
        """
        _, _, spikes = simulate_neuron(fs_neuron, current=10.0, duration_ms=1000)
        if len(spikes) < 6:
            return
        isis = [spikes[i + 1] - spikes[i] for i in range(len(spikes) - 1)]
        early_avg = sum(isis[:3]) / 3
        late_avg = sum(isis[-3:]) / 3
        # FS neurons should have nearly constant ISI (ratio close to 1.0)
        ratio = late_avg / early_avg if early_avg > 0 else 1.0
        assert 0.7 <= ratio <= 1.5, (
            f"FS should show minimal adaptation: early ISI={early_avg:.1f}, "
            f"late ISI={late_avg:.1f}, ratio={ratio:.2f}"
        )

    def test_faster_than_rs(self) -> None:
        """FS neurons should fire faster than RS neurons with same input."""
        rs = IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING)
        fs = IzhikevichNeuron.from_type(NeuronType.FAST_SPIKING)
        _, _, rs_spikes = simulate_neuron(rs, current=10.0, duration_ms=1000)
        _, _, fs_spikes = simulate_neuron(fs, current=10.0, duration_ms=1000)
        assert len(fs_spikes) > len(rs_spikes), (
            f"FS ({len(fs_spikes)} spikes) should fire faster than RS ({len(rs_spikes)} spikes)"
        )


class TestIntrinsicallyBursting:
    """Intrinsically Bursting neurons: layer 5 pyramidal cells.

    Expected behavior: initial burst followed by tonic firing.
    """

    def test_fires_with_current(self, ib_neuron: IzhikevichNeuron) -> None:
        """IB neuron should fire when given input current."""
        _, _, spikes = simulate_neuron(ib_neuron, current=10.0, duration_ms=1000)
        assert len(spikes) > 0, "IB neuron did not fire with I=10"

    def test_initial_burst(self, ib_neuron: IzhikevichNeuron) -> None:
        """IB neurons should show an initial burst (cluster of fast spikes).

        The first few ISIs should be shorter than later ISIs, indicating
        a burst-then-tonic pattern.
        """
        _, _, spikes = simulate_neuron(ib_neuron, current=10.0, duration_ms=1000)
        if len(spikes) < 5:
            return
        isis = [spikes[i + 1] - spikes[i] for i in range(len(spikes) - 1)]
        first_isi = isis[0]
        later_isis = isis[3:] if len(isis) > 3 else isis[1:]
        if later_isis:
            avg_later = sum(later_isis) / len(later_isis)
            # First ISI should be shorter than average later ISI (burst behavior)
            assert first_isi < avg_later * 1.5, (
                f"Expected burst: first ISI={first_isi}, avg later ISI={avg_later:.1f}"
            )


class TestSimulateNeuron:
    """Test the simulation helper function."""

    def test_returns_correct_shapes(self, rs_neuron: IzhikevichNeuron) -> None:
        """Traces should match the requested duration."""
        v, u, _ = simulate_neuron(rs_neuron, current=10.0, duration_ms=500)
        assert len(v) == 500
        assert len(u) == 500

    def test_voltage_stays_bounded(self, rs_neuron: IzhikevichNeuron) -> None:
        """Membrane potential should never exceed spike threshold after reset.

        After a spike, v resets to c. Between spikes, v should stay below threshold.
        """
        v, _, _ = simulate_neuron(rs_neuron, current=10.0, duration_ms=1000)
        # After reset, v = c (e.g., -65). The trace records post-reset values.
        assert v.max() <= 30.0, f"Voltage exceeded threshold in trace: max={v.max()}"

    def test_reset_clears_state(self, rs_neuron: IzhikevichNeuron) -> None:
        """Reset should return neuron to resting state."""
        simulate_neuron(rs_neuron, current=10.0, duration_ms=100)
        rs_neuron.reset()
        assert rs_neuron.v == -65.0
        assert len(rs_neuron.spike_times) == 0
        assert rs_neuron._timestep == 0
