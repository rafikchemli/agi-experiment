"""Experiment 2: Does STDP produce correct weight changes?

Tests verify the fundamental STDP rule:
- Pre fires before post → synapse strengthens (LTP)
- Post fires before pre → synapse weakens (LTD)
- Outside the timing window → no significant change
- Dale's Law is enforced at all times

Reference: Bi & Poo, 1998 STDP timing curves.
Assumptions tested: A3 (asymmetric STDP window)
"""

import math

from brain_sim.synapse import (
    DEFAULT_STDP,
    Synapse,
    SynapseType,
)


class TestSynapseInitialization:
    """Verify synapses start in a valid state with Dale's Law enforced."""

    def test_excitatory_weight_positive(self) -> None:
        """Excitatory synapses must have non-negative weights."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        assert syn.weight >= 0.0

    def test_excitatory_negative_weight_clamped(self) -> None:
        """Negative weight on excitatory synapse should be clamped to 0."""
        syn = Synapse(weight=-0.3, synapse_type=SynapseType.EXCITATORY)
        assert syn.weight >= 0.0

    def test_inhibitory_weight_negative(self) -> None:
        """Inhibitory synapses must have non-positive weights."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.INHIBITORY)
        assert syn.weight <= 0.0

    def test_traces_start_at_zero(self) -> None:
        """All traces should start at zero."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        assert syn.pre_trace == 0.0
        assert syn.post_trace == 0.0
        assert syn.eligibility == 0.0

    def test_default_stdp_params(self) -> None:
        """Default STDP should have a_minus slightly > a_plus for stability."""
        assert DEFAULT_STDP.a_minus > DEFAULT_STDP.a_plus


class TestSTDPBasicRule:
    """Test the core STDP timing rule: pre-before-post → LTP, post-before-pre → LTD."""

    def test_pre_before_post_strengthens(self) -> None:
        """When pre fires before post (causal timing), synapse should strengthen.

        This is the fundamental Hebbian rule: "neurons that fire together wire together."
        Specifically, the PRE neuron predicts the POST neuron's firing.
        """
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        initial_weight = syn.weight

        # Pre fires at t=0
        syn.on_pre_spike()

        # Wait 5ms (traces decay)
        for _ in range(5):
            syn.decay_traces()

        # Post fires at t=5 (pre was before post → LTP)
        syn.on_post_spike()

        assert syn.weight > initial_weight, (
            f"Pre-before-post should strengthen: {initial_weight} -> {syn.weight}"
        )

    def test_post_before_pre_weakens(self) -> None:
        """When post fires before pre (acausal timing), synapse should weaken.

        The presynaptic neuron was NOT causal in the postsynaptic firing.
        """
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        initial_weight = syn.weight

        # Post fires at t=0
        syn.on_post_spike()

        # Wait 5ms
        for _ in range(5):
            syn.decay_traces()

        # Pre fires at t=5 (post was before pre → LTD)
        syn.on_pre_spike()

        assert syn.weight < initial_weight, (
            f"Post-before-pre should weaken: {initial_weight} -> {syn.weight}"
        )

    def test_ltp_magnitude_decreases_with_delay(self) -> None:
        """LTP should be stronger for shorter pre-post delays.

        A spike pair with Δt=2ms should produce more LTP than Δt=15ms.
        This tests the exponential decay of the STDP window.
        """
        # Short delay (2ms)
        syn_short = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn_short.on_pre_spike()
        for _ in range(2):
            syn_short.decay_traces()
        syn_short.on_post_spike()
        dw_short = syn_short.weight - 0.5

        # Long delay (15ms)
        syn_long = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn_long.on_pre_spike()
        for _ in range(15):
            syn_long.decay_traces()
        syn_long.on_post_spike()
        dw_long = syn_long.weight - 0.5

        assert dw_short > dw_long > 0, (
            f"Short delay LTP ({dw_short:.6f}) should exceed long delay LTP ({dw_long:.6f})"
        )

    def test_ltd_magnitude_decreases_with_delay(self) -> None:
        """LTD should be stronger for shorter post-pre delays."""
        # Short delay (2ms)
        syn_short = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn_short.on_post_spike()
        for _ in range(2):
            syn_short.decay_traces()
        syn_short.on_pre_spike()
        dw_short = syn_short.weight - 0.5  # should be negative

        # Long delay (15ms)
        syn_long = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn_long.on_post_spike()
        for _ in range(15):
            syn_long.decay_traces()
        syn_long.on_pre_spike()
        dw_long = syn_long.weight - 0.5

        assert dw_short < dw_long < 0, (
            f"Short delay LTD ({dw_short:.6f}) should be more negative than "
            f"long delay LTD ({dw_long:.6f})"
        )

    def test_no_change_outside_window(self) -> None:
        """Spikes far apart (>5 tau) should produce negligible weight change."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn.on_pre_spike()

        # Wait 200ms (10x tau) — trace should be ~0
        for _ in range(200):
            syn.decay_traces()

        syn.on_post_spike()
        dw = abs(syn.weight - 0.5)
        assert dw < 1e-6, f"Weight changed by {dw} for spikes 200ms apart (should be ~0)"


class TestTraceDecay:
    """Verify that spike traces decay exponentially with the correct time constant."""

    def test_pre_trace_decays_exponentially(self) -> None:
        """Pre trace should follow exp(-t/tau_plus)."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn.on_pre_spike()  # trace = 1.0

        # After tau_plus ms, trace should be ~1/e ≈ 0.368
        for _ in range(int(DEFAULT_STDP.tau_plus)):
            syn.decay_traces()

        expected = math.exp(-1.0)  # e^(-tau/tau) = e^-1
        assert abs(syn.pre_trace - expected) < 0.01, (
            f"Pre trace after 1 tau: {syn.pre_trace:.4f}, expected ~{expected:.4f}"
        )

    def test_post_trace_decays_exponentially(self) -> None:
        """Post trace should follow exp(-t/tau_minus)."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn.on_post_spike()

        for _ in range(int(DEFAULT_STDP.tau_minus)):
            syn.decay_traces()

        expected = math.exp(-1.0)
        assert abs(syn.post_trace - expected) < 0.01, (
            f"Post trace after 1 tau: {syn.post_trace:.4f}, expected ~{expected:.4f}"
        )

    def test_trace_accumulates_on_repeated_spikes(self) -> None:
        """Multiple spikes should stack traces (temporal summation)."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn.on_pre_spike()  # trace = 1.0
        for _ in range(5):
            syn.decay_traces()
        syn.on_pre_spike()  # trace = decayed_trace + 1.0

        assert syn.pre_trace > 1.0, f"Repeated spikes should stack: trace={syn.pre_trace:.4f}"


class TestDalesLaw:
    """Dale's Law: excitatory synapses stay positive, inhibitory stay negative."""

    def test_excitatory_cannot_go_negative(self) -> None:
        """Even with strong LTD, excitatory weight must not go below w_min."""
        syn = Synapse(weight=0.01, synapse_type=SynapseType.EXCITATORY)

        # Force heavy LTD: post fires, then pre fires immediately
        for _ in range(50):
            syn.on_post_spike()
            syn.decay_traces()
            syn.on_pre_spike()
            syn.decay_traces()

        assert syn.weight >= 0.0, f"Excitatory weight went negative: {syn.weight}"

    def test_inhibitory_cannot_go_positive(self) -> None:
        """Even with strong LTP, inhibitory weight must not go above -w_min."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.INHIBITORY)

        # Force heavy LTP: pre fires, then post fires immediately
        for _ in range(50):
            syn.on_pre_spike()
            syn.decay_traces()
            syn.on_post_spike()
            syn.decay_traces()

        assert syn.weight <= 0.0, f"Inhibitory weight went positive: {syn.weight}"

    def test_excitatory_bounded_by_w_max(self) -> None:
        """Excitatory weight should not exceed w_max."""
        syn = Synapse(weight=0.9, synapse_type=SynapseType.EXCITATORY)

        # Force heavy LTP
        for _ in range(100):
            syn.on_pre_spike()
            syn.on_post_spike()
            syn.decay_traces()

        assert syn.weight <= syn.stdp.w_max, f"Weight {syn.weight} exceeded w_max {syn.stdp.w_max}"


class TestEligibilityTrace:
    """Test the eligibility trace for three-factor learning (Experiment 6 prep)."""

    def test_stdp_accumulates_in_eligibility_when_not_applied(self) -> None:
        """With apply_stdp=False, changes go to eligibility trace, not weight."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        initial_weight = syn.weight

        syn.on_pre_spike(apply_stdp=False)
        for _ in range(5):
            syn.decay_traces()
        syn.on_post_spike(apply_stdp=False)

        # Weight should NOT have changed
        assert syn.weight == initial_weight, "Weight changed when apply_stdp=False"
        # Eligibility should have accumulated
        assert syn.eligibility != 0.0, "Eligibility should be non-zero"

    def test_reward_applies_eligibility(self) -> None:
        """Positive reward should apply accumulated eligibility to weight."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)

        # Build up positive eligibility (pre-before-post → positive STDP)
        syn.on_pre_spike(apply_stdp=False)
        for _ in range(3):
            syn.decay_traces()
        syn.on_post_spike(apply_stdp=False)

        assert syn.eligibility > 0.0
        initial_weight = syn.weight

        # Apply positive reward
        syn.apply_reward(reward=1.0)
        assert syn.weight > initial_weight, (
            f"Positive reward should increase weight: {initial_weight} -> {syn.weight}"
        )

    def test_negative_reward_weakens(self) -> None:
        """Negative reward on positive eligibility should weaken synapse."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)

        syn.on_pre_spike(apply_stdp=False)
        for _ in range(3):
            syn.decay_traces()
        syn.on_post_spike(apply_stdp=False)

        initial_weight = syn.weight
        syn.apply_reward(reward=-1.0)
        assert syn.weight < initial_weight, (
            f"Negative reward should decrease weight: {initial_weight} -> {syn.weight}"
        )


class TestConductionDelay:
    """Verify spike delivery respects conduction delay."""

    def test_spike_delayed_by_correct_amount(self) -> None:
        """A spike queued at t=0 with delay=3 should deliver at t=3."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY, delay=3)
        syn.deliver_spike(timestep=0)

        assert not syn.check_delivery(timestep=0)
        assert not syn.check_delivery(timestep=1)
        assert not syn.check_delivery(timestep=2)
        assert syn.check_delivery(timestep=3)

    def test_default_delay_is_1ms(self) -> None:
        """Default delay should be 1ms."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        assert syn.delay == 1

    def test_multiple_spikes_queued(self) -> None:
        """Multiple spikes should be delivered in order."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY, delay=2)
        syn.deliver_spike(timestep=0)  # delivers at t=2
        syn.deliver_spike(timestep=1)  # delivers at t=3

        assert not syn.check_delivery(timestep=1)
        assert syn.check_delivery(timestep=2)
        assert syn.check_delivery(timestep=3)


class TestSynapseReset:
    """Test that reset clears traces but preserves weight."""

    def test_reset_clears_traces(self) -> None:
        """Reset should zero out traces and eligibility."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn.on_pre_spike()
        syn.on_post_spike()
        syn.eligibility = 0.5

        syn.reset()
        assert syn.pre_trace == 0.0
        assert syn.post_trace == 0.0
        assert syn.eligibility == 0.0

    def test_reset_preserves_weight(self) -> None:
        """Reset should NOT change the learned weight."""
        syn = Synapse(weight=0.5, synapse_type=SynapseType.EXCITATORY)
        syn.on_pre_spike()
        syn.on_post_spike()  # weight changes via STDP

        weight_after_learning = syn.weight
        syn.reset()
        assert syn.weight == weight_after_learning
