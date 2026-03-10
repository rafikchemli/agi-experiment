"""Synapse model with Spike-Timing-Dependent Plasticity (STDP).

A biological synapse is NOT just a scalar weight. It has:
- A strength (weight) that changes via STDP
- Pre/post spike traces for computing timing-dependent plasticity
- An eligibility trace for three-factor learning (reward modulation)
- Dale's Law enforcement: excitatory synapses stay positive, inhibitory stay negative

STDP rule (Bi & Poo, 1998):
- Pre fires before post (causal, Δt > 0) → LTP (strengthen)
- Post fires before pre (acausal, Δt < 0) → LTD (weaken)
- Magnitude decays exponentially with |Δt|

Reference: Bi, G.Q. & Poo, M.M. (1998). Synaptic modifications in cultured
hippocampal neurons. J Neurosci, 18(24), 10464-10472.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class SynapseType(StrEnum):
    """Synapse polarity — enforced by Dale's Law."""

    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"


class STDPParams(BaseModel):
    """Parameters for the STDP learning rule.

    These are 'genetic' parameters — set by the neuron type, not learned.

    Args:
        a_plus: LTP amplitude (pre-before-post strengthening).
        a_minus: LTD amplitude (post-before-pre weakening).
        tau_plus: LTP time constant in ms.
        tau_minus: LTD time constant in ms.
        w_max: Maximum absolute weight (prevents runaway).
        w_min: Minimum absolute weight (prevents complete silencing).
    """

    a_plus: float = Field(default=0.01, description="LTP amplitude")
    a_minus: float = Field(
        default=0.012, description="LTD amplitude (slightly > a_plus for stability)"
    )
    tau_plus: float = Field(default=20.0, description="LTP time constant (ms)")
    tau_minus: float = Field(default=20.0, description="LTD time constant (ms)")
    w_max: float = Field(default=1.0, description="Maximum absolute weight")
    w_min: float = Field(default=0.0, description="Minimum absolute weight")


# Default STDP parameters based on Bi & Poo 1998 experimental data
DEFAULT_STDP = STDPParams()


class Synapse:
    """A biological synapse with STDP plasticity.

    Tracks pre/post spike traces and computes weight updates based on
    spike timing. Enforces Dale's Law: excitatory weights stay >= 0,
    inhibitory weights stay <= 0.

    The eligibility trace stores STDP-computed weight changes that haven't
    been applied yet — they wait for a neuromodulatory signal (dopamine)
    to gate the actual weight update. This is the three-factor learning rule.

    Args:
        weight: Initial synaptic strength.
        synapse_type: Excitatory or inhibitory (Dale's Law).
        stdp_params: STDP learning rule parameters.
        delay: Conduction delay in ms (signal travel time).
    """

    def __init__(
        self,
        weight: float,
        synapse_type: SynapseType,
        stdp_params: STDPParams | None = None,
        delay: int = 1,
        plastic: bool = True,
    ) -> None:
        self.synapse_type = synapse_type
        self.stdp = stdp_params or DEFAULT_STDP
        self.delay = delay
        self.plastic = plastic

        # Enforce Dale's Law at initialization
        if synapse_type == SynapseType.EXCITATORY:
            self.weight = max(0.0, weight)
        else:
            self.weight = min(0.0, -abs(weight))

        # STDP traces — exponentially decaying memory of recent spikes
        self.pre_trace: float = 0.0
        self.post_trace: float = 0.0

        # Eligibility trace — for three-factor learning (Exp 6)
        self.eligibility: float = 0.0

        # Spike delivery buffer (for conduction delay)
        self._spike_buffer: list[int] = []

    @property
    def abs_weight(self) -> float:
        """Absolute weight magnitude."""
        return abs(self.weight)

    def on_pre_spike(self, apply_stdp: bool = True) -> float:
        """Called when the presynaptic neuron fires.

        Updates the pre_trace and computes LTD (weakening) based on
        the current post_trace (if post fired recently before pre, weaken).

        Args:
            apply_stdp: Whether to apply weight change directly (True for
                unsupervised layers) or accumulate in eligibility trace
                (False for reward-modulated layers).

        Returns:
            The STDP weight change computed (before application).
        """
        # LTD: post fired before pre → weaken
        dw = -self.stdp.a_minus * self.post_trace

        if apply_stdp:
            self._apply_weight_change(dw)
        else:
            self.eligibility += dw

        # Update pre trace
        self.pre_trace += 1.0

        return dw

    def on_post_spike(self, apply_stdp: bool = True) -> float:
        """Called when the postsynaptic neuron fires.

        Updates the post_trace and computes LTP (strengthening) based on
        the current pre_trace (if pre fired recently before post, strengthen).

        Args:
            apply_stdp: Whether to apply weight change directly or accumulate
                in eligibility trace.

        Returns:
            The STDP weight change computed (before application).
        """
        # LTP: pre fired before post → strengthen
        dw = self.stdp.a_plus * self.pre_trace

        if apply_stdp:
            self._apply_weight_change(dw)
        else:
            self.eligibility += dw

        # Update post trace
        self.post_trace += 1.0

        return dw

    def decay_traces(self, dt: float = 1.0) -> None:
        """Decay spike traces by one timestep.

        Called every simulation tick. Traces decay exponentially.

        Args:
            dt: Timestep in ms.
        """
        import math

        self.pre_trace *= math.exp(-dt / self.stdp.tau_plus)
        self.post_trace *= math.exp(-dt / self.stdp.tau_minus)

    def apply_reward(self, reward: float, learning_rate: float = 1.0) -> float:
        """Apply reward-modulated weight change from eligibility trace.

        Three-factor learning: weight change = eligibility × reward × lr.
        Called when a neuromodulatory signal (dopamine) arrives.

        Args:
            reward: Reward signal (+1 correct, -1 incorrect, 0 neutral).
            learning_rate: Scaling factor for the update.

        Returns:
            The actual weight change applied.
        """
        dw = learning_rate * self.eligibility * reward
        self._apply_weight_change(dw)
        self.eligibility *= 0.9  # decay eligibility after use
        return dw

    def deliver_spike(self, timestep: int) -> None:
        """Queue a spike for delivery after conduction delay.

        Args:
            timestep: Current simulation timestep.
        """
        self._spike_buffer.append(timestep + self.delay)

    def check_delivery(self, timestep: int) -> bool:
        """Check if a delayed spike should be delivered this timestep.

        Args:
            timestep: Current simulation timestep.

        Returns:
            True if a spike should be delivered now.
        """
        if self._spike_buffer and self._spike_buffer[0] <= timestep:
            self._spike_buffer.pop(0)
            return True
        return False

    def get_current(self) -> float:
        """Get the synaptic current to deliver to the postsynaptic neuron.

        Returns:
            Current proportional to the weight. Positive for excitatory,
            negative for inhibitory.
        """
        return self.weight

    def _apply_weight_change(self, dw: float) -> None:
        """Apply a weight change while enforcing Dale's Law and bounds.

        Args:
            dw: Raw weight change to apply.
        """
        self.weight += dw

        # Enforce Dale's Law: sign must match synapse type
        if self.synapse_type == SynapseType.EXCITATORY:
            self.weight = max(self.stdp.w_min, min(self.stdp.w_max, self.weight))
        else:
            self.weight = max(-self.stdp.w_max, min(-self.stdp.w_min, self.weight))

    def reset(self) -> None:
        """Reset synapse to initial state (traces only, not weight)."""
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.eligibility = 0.0
        self._spike_buffer = []
