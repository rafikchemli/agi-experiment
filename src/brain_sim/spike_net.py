"""Vectorized spiking network — designed for scale.

The original Network class (network.py) uses Python loops over individual
neurons and synapses. This works for small circuits (Experiments 1-5) but
won't scale to the hundreds of neurons needed for Experiments 6-7.

This module provides the same biological semantics with numpy-vectorized
internals. The core operations — spike propagation, STDP, PSC decay — become
matrix operations instead of Python loops.

Design philosophy (inspired by what makes transformers scalable):
1. Fixed computational pattern: spike propagation = W.T @ spike_vector
2. Composable layers: Population + Projection = composable building blocks
3. Vectorized state: all membrane potentials in one array, not N objects
4. Clear API: add populations, connect with projections, step the network

Performance: O(N^2) for dense all-to-all, O(K) for sparse — but the constant
factor is ~100x smaller than Python loops because numpy uses BLAS.
"""

import math

import numpy as np

from brain_sim.neuron import NEURON_PRESETS, NeuronType
from brain_sim.synapse import DEFAULT_STDP, STDPParams, SynapseType

SPIKE_THRESHOLD: float = 30.0
RESTING_V: float = -65.0


class Population:
    """A group of neurons of the same type, stored as numpy arrays.

    All neurons in a population share the same Izhikevich parameters (a, b, c, d)
    — their "genetic code." State (v, u) is per-neuron and evolves independently.

    This is the vectorized equivalent of N individual IzhikevichNeuron objects.

    Args:
        n: Number of neurons in this population.
        neuron_type: Biological type (determines Izhikevich parameters).
        label: Human-readable name for this population.
    """

    def __init__(self, n: int, neuron_type: NeuronType, label: str) -> None:
        params = NEURON_PRESETS[neuron_type]
        self.n = n
        self.label = label
        self.neuron_type = neuron_type

        # Izhikevich parameters — same for all neurons in population
        self.a = params.a
        self.b = params.b
        self.c = params.c
        self.d = params.d

        # Vectorized state
        self.v = np.full(n, RESTING_V, dtype=np.float64)
        self.u = np.full(n, params.b * RESTING_V, dtype=np.float64)

        # Post-synaptic current accumulator (set by projections each step)
        self.psc = np.zeros(n, dtype=np.float64)

        # Spike output
        self.fired = np.zeros(n, dtype=bool)
        self.spike_counts = np.zeros(n, dtype=np.int64)

        # Spike log: timestep -> indices of neurons that fired
        self._spike_log: dict[int, np.ndarray] = {}
        self._timestep = 0

    def step(self, external_currents: np.ndarray | None = None) -> np.ndarray:
        """Advance all neurons by one timestep (1ms).

        Uses 0.5ms sub-steps for numerical stability (Izhikevich recommendation).

        Args:
            external_currents: Per-neuron external input. Shape (n,).

        Returns:
            Boolean array of which neurons fired.
        """
        total_current = self.psc.copy()
        if external_currents is not None:
            total_current += external_currents

        v = self.v.copy()
        u = self.u.copy()

        # Two 0.5ms sub-steps
        for _ in range(2):
            not_spiked = v < SPIKE_THRESHOLD
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + total_current) * 0.5
            v = np.where(not_spiked, v + dv, v)
            du = self.a * (self.b * v - u) * 0.5
            u = np.where(not_spiked, u + du, u)

        # Spike detection and reset
        self.fired = v >= SPIKE_THRESHOLD
        v[self.fired] = self.c
        u[self.fired] = u[self.fired] + self.d

        self.v = v
        self.u = u

        # Log spikes
        if self.fired.any():
            self.spike_counts += self.fired.astype(np.int64)
            self._spike_log[self._timestep] = np.where(self.fired)[0]

        self._timestep += 1
        return self.fired

    def recent_spike_count(self, window_ms: int) -> np.ndarray:
        """Count spikes per neuron in the last window_ms milliseconds.

        Args:
            window_ms: How far back to look.

        Returns:
            Array of shape (n,) with spike counts per neuron.
        """
        counts = np.zeros(self.n, dtype=np.int64)
        cutoff = self._timestep - window_ms
        for t, indices in self._spike_log.items():
            if t > cutoff:
                counts[indices] += 1
        return counts

    def reset(self) -> None:
        """Reset all state (membrane potential, recovery, spikes)."""
        self.v[:] = RESTING_V
        self.u[:] = self.b * RESTING_V
        self.psc[:] = 0.0
        self.fired[:] = False
        self.spike_counts[:] = 0
        self._spike_log.clear()
        self._timestep = 0


class Projection:
    """Synaptic connections between two populations as a weight matrix.

    The weight matrix W has shape (pre.n, post.n). Entry W[i,j] is the
    weight from pre neuron i to post neuron j.

    Current delivery: post.psc += W.T @ pre_spike_vector
    This is a single matrix multiply — O(pre.n × post.n) but vectorized.

    Args:
        pre: Source population.
        post: Target population.
        weights: Weight matrix of shape (pre.n, post.n).
        synapse_type: Excitatory or inhibitory (Dale's Law).
        plastic: Whether STDP is active for this projection.
        stdp_params: STDP parameters (uses defaults if None).
    """

    def __init__(
        self,
        pre: Population,
        post: Population,
        weights: np.ndarray,
        synapse_type: SynapseType,
        plastic: bool = True,
        stdp_params: STDPParams | None = None,
        reward_modulated: bool = False,
    ) -> None:
        if weights.shape != (pre.n, post.n):
            msg = f"Weight shape {weights.shape} does not match ({pre.n}, {post.n})"
            raise ValueError(msg)

        self.pre = pre
        self.post = post
        self.synapse_type = synapse_type
        self.plastic = plastic
        self.reward_modulated = reward_modulated
        self.stdp = stdp_params or DEFAULT_STDP

        # Weight matrix — enforce Dale's Law at initialization
        self.weights = weights.copy()
        self._enforce_bounds()

        # STDP traces (one per neuron, not per synapse — efficient)
        self.pre_traces = np.zeros(pre.n, dtype=np.float64)
        self.post_traces = np.zeros(post.n, dtype=np.float64)

        # Eligibility traces for three-factor learning
        self.eligibility = np.zeros_like(self.weights)

    def deliver_and_ltd(self, pre_arriving: np.ndarray, apply_stdp: bool = True) -> None:
        """Deliver current from arriving pre spikes and compute LTD.

        This is phase 1 of the biological STDP cycle:
        pre spike arrives → deliver current → update pre_trace → LTD.

        In biology, this happens when the presynaptic spike physically arrives
        at the synapse (after conduction delay).

        Args:
            pre_arriving: Boolean array of pre neurons whose spikes arrive now.
            apply_stdp: Whether to apply weight change directly.
        """
        if not pre_arriving.any():
            return

        # Deliver current
        spike_vec = pre_arriving.astype(np.float64)
        self.post.psc += self.weights.T @ spike_vec

        if not self.plastic:
            return

        # LTD: pre arrives, post trace tells us if post fired recently → weaken
        dw = -self.stdp.a_minus * np.outer(spike_vec, self.post_traces)
        if apply_stdp:
            self.weights += dw
        else:
            self.eligibility += dw

        # Update pre traces (arriving pre neurons get +1)
        self.pre_traces[pre_arriving] += 1.0

    def on_post_spike(self, post_fired: np.ndarray, apply_stdp: bool = True) -> None:
        """Compute LTP for post neurons that just fired.

        This is phase 2 of the biological STDP cycle:
        post neuron fires → update post_trace → LTP from pre_trace.

        Args:
            post_fired: Boolean array of post neurons that just fired.
            apply_stdp: Whether to apply weight change directly.
        """
        if not post_fired.any():
            return

        if self.plastic:
            # LTP: post fires, pre_trace tells us if pre fired recently → strengthen
            dw = self.stdp.a_plus * np.outer(
                self.pre_traces,
                post_fired.astype(np.float64),
            )
            if apply_stdp:
                self.weights += dw
            else:
                self.eligibility += dw

        # Update post traces
        self.post_traces[post_fired] += 1.0

        # Enforce bounds
        if apply_stdp and self.plastic:
            self._enforce_bounds()

    def decay_traces(self) -> None:
        """Decay STDP traces by one timestep (exponential decay)."""
        self.pre_traces *= math.exp(-1.0 / self.stdp.tau_plus)
        self.post_traces *= math.exp(-1.0 / self.stdp.tau_minus)

    def apply_reward(self, reward: float, learning_rate: float = 1.0) -> None:
        """Apply reward-modulated weight change from eligibility traces.

        Three-factor learning: dw = eligibility × reward × lr.

        Args:
            reward: Reward signal (+1 correct, -1 incorrect, 0 neutral).
            learning_rate: Scaling factor.
        """
        self.weights += learning_rate * self.eligibility * reward
        self.eligibility *= 0.9  # decay eligibility after use
        self._enforce_bounds()

    def _enforce_bounds(self) -> None:
        """Enforce weight bounds and Dale's Law (vectorized)."""
        if self.synapse_type == SynapseType.EXCITATORY:
            np.clip(
                self.weights,
                self.stdp.w_min,
                self.stdp.w_max,
                out=self.weights,
            )
        else:
            np.clip(
                self.weights,
                -self.stdp.w_max,
                -self.stdp.w_min,
                out=self.weights,
            )

    def reset(self) -> None:
        """Reset traces (not weights)."""
        self.pre_traces[:] = 0.0
        self.post_traces[:] = 0.0
        self.eligibility[:] = 0.0


class SpikeNetwork:
    """A network of populations connected by projections.

    Composable design: add populations and projections, then step the whole
    network. Each step:
    1. Decay PSC for all populations
    2. Deliver current from pre spikes to post neurons (matrix multiply)
    3. Step all neurons (vectorized Izhikevich ODE)
    4. Update STDP for plastic projections
    5. Decay STDP traces

    Args:
        tau_syn: Post-synaptic current decay time constant (ms).
        plasticity: Global plasticity switch (can be overridden per projection).
    """

    def __init__(
        self,
        tau_syn: float = 5.0,
        plasticity: bool = True,
    ) -> None:
        self.tau_syn = tau_syn
        self.plasticity = plasticity
        self.populations: dict[str, Population] = {}
        self.projections: list[Projection] = []
        self._timestep = 0

    def add_population(self, pop: Population) -> None:
        """Register a population with the network.

        Args:
            pop: The population to add.

        Raises:
            ValueError: If a population with this label already exists.
        """
        if pop.label in self.populations:
            msg = f"Population '{pop.label}' already exists"
            raise ValueError(msg)
        self.populations[pop.label] = pop

    def add_projection(self, proj: Projection) -> None:
        """Register a projection with the network.

        Args:
            proj: The projection to add.
        """
        self.projections.append(proj)

    def step(
        self,
        external_currents: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Advance the entire network by one timestep.

        The step sequence matches biological spike timing:
        1. Decay PSC (receptor kinetics)
        2. Deliver arriving pre spikes + LTD (pre spike arrival phase)
        3. Step all neurons (membrane potential update)
        4. LTP for post neurons that fired (post spike phase)
        5. Decay STDP traces

        Conduction delay: spikes from timestep t arrive at t+1.
        We achieve this by using the _prev_fired snapshots.

        Args:
            external_currents: Dict mapping population label to current array.

        Returns:
            Dict mapping population label to boolean fired array.
        """
        ext = external_currents or {}
        decay_factor = math.exp(-1.0 / self.tau_syn)

        # 1. Decay PSC
        for pop in self.populations.values():
            pop.psc *= decay_factor

        # 2. Deliver arriving spikes (from PREVIOUS timestep) + LTD
        # pre.fired still holds last step's result (step() hasn't been called yet)
        # For reward_modulated projections, STDP accumulates in eligibility
        # instead of being applied directly (three-factor learning).
        for proj in self.projections:
            apply = self.plasticity and proj.plastic and not proj.reward_modulated
            proj.deliver_and_ltd(proj.pre.fired, apply_stdp=apply)

        # 3. Step all populations (updates fired to NEW values)
        fired: dict[str, np.ndarray] = {}
        for label, pop in self.populations.items():
            pop.step(ext.get(label))
            fired[label] = pop.fired.copy()

        # 4. LTP for post neurons that just fired
        if self.plasticity:
            for proj in self.projections:
                apply = proj.plastic and not proj.reward_modulated
                proj.on_post_spike(proj.post.fired, apply_stdp=apply)

        # 5. Decay STDP traces
        for proj in self.projections:
            proj.decay_traces()

        self._timestep += 1
        return fired

    def simulate(
        self,
        duration_ms: int,
        external_currents: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the network for a fixed duration with constant input.

        Args:
            duration_ms: Simulation duration in milliseconds.
            external_currents: Constant external current per population.

        Returns:
            Dict mapping population label to spike count array.
        """
        for _ in range(duration_ms):
            self.step(external_currents)
        return {label: pop.spike_counts.copy() for label, pop in self.populations.items()}

    def reset(self) -> None:
        """Reset all populations and projections."""
        for pop in self.populations.values():
            pop.reset()
        for proj in self.projections:
            proj.reset()
        self._timestep = 0


def build_v1_circuit_vectorized(
    grid_size: int = 8,
    n_v1_excitatory: int = 4,
    n_v1_inhibitory: int = 2,
    w_input_to_v1: float = 3.0,
    w_v1_to_inh: float = 5.0,
    w_inh_to_v1: float = 15.0,
    inh_tonic: float = 3.0,
    tau_syn: float = 5.0,
    stdp_params: STDPParams | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SpikeNetwork, dict[str, Population], float]:
    """Build a retina -> V1 circuit using vectorized populations.

    Same architecture as layers.build_v1_circuit but using the vectorized
    SpikeNetwork for performance.

    Args:
        grid_size: Input image size (grid_size x grid_size pixels).
        n_v1_excitatory: Number of V1 excitatory neurons.
        n_v1_inhibitory: Number of V1 inhibitory neurons.
        w_input_to_v1: Mean initial weight for input->V1 synapses.
        w_v1_to_inh: Weight for V1->inhibitory synapses.
        w_inh_to_v1: Weight for inhibitory->V1 synapses.
        inh_tonic: Tonic current for inhibitory neurons.
        tau_syn: PSC decay time constant (ms).
        stdp_params: STDP parameters for input->V1 synapses.
        rng: NumPy random generator for reproducible weight initialization.

    Returns:
        Tuple of (network, populations_dict, inh_tonic).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if stdp_params is None:
        stdp_params = STDPParams(
            a_plus=0.005,
            a_minus=0.006,
            tau_plus=20.0,
            tau_minus=20.0,
            w_max=10.0,
            w_min=0.0,
        )

    n_pixels = grid_size * grid_size

    # Create populations
    retina = Population(n_pixels, NeuronType.REGULAR_SPIKING, "retina")
    v1_exc = Population(n_v1_excitatory, NeuronType.REGULAR_SPIKING, "v1_exc")
    v1_inh = Population(n_v1_inhibitory, NeuronType.FAST_SPIKING, "v1_inh")

    # Create network
    net = SpikeNetwork(tau_syn=tau_syn, plasticity=True)
    net.add_population(retina)
    net.add_population(v1_exc)
    net.add_population(v1_inh)

    # Input -> V1 exc: all-to-all, random initial weights, STDP plastic
    w_in = rng.normal(w_input_to_v1, w_input_to_v1 * 0.3, (n_pixels, n_v1_excitatory))
    np.clip(w_in, stdp_params.w_min, stdp_params.w_max, out=w_in)
    net.add_projection(
        Projection(
            retina,
            v1_exc,
            w_in,
            SynapseType.EXCITATORY,
            plastic=True,
            stdp_params=stdp_params,
        )
    )

    # V1 exc -> V1 inh: all-to-all, fixed
    w_ei = np.full((n_v1_excitatory, n_v1_inhibitory), w_v1_to_inh)
    net.add_projection(
        Projection(
            v1_exc,
            v1_inh,
            w_ei,
            SynapseType.EXCITATORY,
            plastic=False,
        )
    )

    # V1 inh -> V1 exc: all-to-all, fixed
    w_ie = np.full((n_v1_inhibitory, n_v1_excitatory), w_inh_to_v1)
    net.add_projection(
        Projection(
            v1_inh,
            v1_exc,
            w_ie,
            SynapseType.INHIBITORY,
            plastic=False,
        )
    )

    pops = {"retina": retina, "v1_exc": v1_exc, "v1_inh": v1_inh}
    return net, pops, inh_tonic
