"""Network assembly and clock-driven simulation.

A biological neural network is a dynamical system running on a clock.
Every millisecond:
1. Post-synaptic currents decay (exponential, mimicking receptor kinetics)
2. Delayed spikes arrive and add to post-synaptic currents
3. All neurons update their membrane potential
4. Fired neurons propagate spikes through outgoing synapses
5. STDP traces update

Key biological feature: post-synaptic currents (PSCs) decay exponentially,
not instantaneously. This allows temporal summation of inputs — critical
for circuit dynamics like Winner-Take-All.

PSC time constants: AMPA ~2-5ms (excitatory), GABA-A ~5-10ms (inhibitory).
We use a single tau_syn for simplicity (Assumption: can split later if needed).
"""

import math
from dataclasses import dataclass

from brain_sim.neuron import IzhikevichNeuron, NeuronType
from brain_sim.synapse import Synapse, SynapseType


@dataclass
class Connection:
    """A directed synaptic connection between two neurons.

    Attributes:
        pre_id: Presynaptic neuron ID.
        post_id: Postsynaptic neuron ID.
        synapse: The synapse object managing weight and STDP.
    """

    pre_id: int
    post_id: int
    synapse: Synapse


class Network:
    """A network of spiking neurons connected by synapses.

    Clock-driven at 1ms resolution. Tracks post-synaptic currents with
    exponential decay for realistic temporal summation.

    Args:
        plasticity: If True, STDP learning is active. If False, weights
            are frozen (useful for testing circuit dynamics).
        tau_syn: Post-synaptic current decay time constant in ms.
    """

    def __init__(self, plasticity: bool = True, tau_syn: float = 5.0) -> None:
        self.plasticity = plasticity
        self.tau_syn = tau_syn
        self.neurons: dict[int, IzhikevichNeuron] = {}
        self._connections: list[Connection] = []
        self._outgoing: dict[int, list[int]] = {}
        self._incoming: dict[int, list[int]] = {}
        self._psc: dict[int, float] = {}
        self._timestep: int = 0
        self._spike_log: dict[int, list[int]] = {}

    def add_neuron(self, neuron_id: int, neuron: IzhikevichNeuron) -> None:
        """Add a neuron to the network.

        Args:
            neuron_id: Unique integer ID for this neuron.
            neuron: The neuron object.

        Raises:
            ValueError: If neuron_id already exists.
        """
        if neuron_id in self.neurons:
            msg = f"Neuron {neuron_id} already exists"
            raise ValueError(msg)
        self.neurons[neuron_id] = neuron
        self._outgoing[neuron_id] = []
        self._incoming[neuron_id] = []
        self._psc[neuron_id] = 0.0
        self._spike_log[neuron_id] = []

    def connect(self, pre_id: int, post_id: int, synapse: Synapse) -> None:
        """Connect two neurons with a synapse.

        Args:
            pre_id: Presynaptic neuron ID.
            post_id: Postsynaptic neuron ID.
            synapse: The synapse object.

        Raises:
            ValueError: If either neuron ID is not in the network.
        """
        if pre_id not in self.neurons:
            msg = f"Pre neuron {pre_id} not in network"
            raise ValueError(msg)
        if post_id not in self.neurons:
            msg = f"Post neuron {post_id} not in network"
            raise ValueError(msg)
        idx = len(self._connections)
        self._connections.append(Connection(pre_id=pre_id, post_id=post_id, synapse=synapse))
        self._outgoing[pre_id].append(idx)
        self._incoming[post_id].append(idx)

    def step(self, external_currents: dict[int, float] | None = None) -> list[int]:
        """Advance the network by one timestep (1ms).

        Args:
            external_currents: External input current per neuron ID.

        Returns:
            List of neuron IDs that fired this timestep.
        """
        currents = external_currents or {}
        decay_factor = math.exp(-1.0 / self.tau_syn)

        # 1. Decay post-synaptic currents
        for nid in self.neurons:
            self._psc[nid] *= decay_factor

        # 2. Deliver arriving spikes (after conduction delay)
        for conn in self._connections:
            if conn.synapse.check_delivery(self._timestep):
                apply = self.plasticity and conn.synapse.plastic
                conn.synapse.on_pre_spike(apply_stdp=apply)
                self._psc[conn.post_id] += conn.synapse.get_current()

        # 3. Step all neurons with total current (PSC + external)
        fired: list[int] = []
        for nid, neuron in self.neurons.items():
            total_current = self._psc[nid] + currents.get(nid, 0.0)
            spiked = neuron.step(current=total_current)
            if spiked:
                fired.append(nid)
                self._spike_log[nid].append(self._timestep)

        # 4. Propagate spikes and update STDP post traces
        for nid in fired:
            for conn_idx in self._outgoing[nid]:
                conn = self._connections[conn_idx]
                conn.synapse.deliver_spike(self._timestep)
            for conn_idx in self._incoming[nid]:
                conn = self._connections[conn_idx]
                apply = self.plasticity and conn.synapse.plastic
                conn.synapse.on_post_spike(apply_stdp=apply)

        # 5. Decay STDP traces
        for conn in self._connections:
            conn.synapse.decay_traces()

        self._timestep += 1
        return fired

    def simulate(
        self,
        duration_ms: int,
        external_currents: dict[int, float] | None = None,
    ) -> dict[int, list[int]]:
        """Run the network for a fixed duration with constant input.

        Args:
            duration_ms: Simulation duration in milliseconds.
            external_currents: Constant external current per neuron.

        Returns:
            Dict mapping neuron ID to list of spike timesteps.
        """
        for _ in range(duration_ms):
            self.step(external_currents)
        return {nid: list(spikes) for nid, spikes in self._spike_log.items()}

    def get_spike_counts(self) -> dict[int, int]:
        """Get total spike count per neuron.

        Returns:
            Dict mapping neuron ID to spike count.
        """
        return {nid: len(spikes) for nid, spikes in self._spike_log.items()}

    def incoming_connections(self, neuron_id: int) -> list[Connection]:
        """Get all connections incoming to a neuron.

        Args:
            neuron_id: The postsynaptic neuron ID.

        Returns:
            List of Connection objects targeting this neuron.
        """
        return [self._connections[idx] for idx in self._incoming[neuron_id]]

    def recent_spike_count(self, neuron_id: int, window_ms: int) -> int:
        """Count spikes in the last window_ms milliseconds.

        Args:
            neuron_id: Neuron to query.
            window_ms: How far back to look (in ms).

        Returns:
            Number of spikes in the window.
        """
        cutoff = self._timestep - window_ms
        return sum(1 for t in self._spike_log.get(neuron_id, []) if t > cutoff)

    def reset(self) -> None:
        """Reset all neurons, synapses, and spike logs."""
        for neuron in self.neurons.values():
            neuron.reset()
        for conn in self._connections:
            conn.synapse.reset()
        self._psc = {nid: 0.0 for nid in self.neurons}
        self._timestep = 0
        self._spike_log = {nid: [] for nid in self.neurons}


def build_wta_circuit(
    n_excitatory: int = 5,
    n_inhibitory: int = 1,
    w_exc_to_inh: float = 5.0,
    w_inh_to_exc: float = 8.0,
    plasticity: bool = False,
    tau_syn: float = 5.0,
) -> Network:
    """Build a Winner-Take-All circuit with lateral inhibition.

    Architecture: excitatory neurons -> shared inhibitory pool -> back to
    all excitatory neurons. This creates competition: active E neurons
    drive I, and I suppresses all E neurons. The strongest-driven E neuron
    overcomes the inhibition; weaker ones are suppressed.

    This is a canonical cortical microcircuit observed across sensory areas.

    Neuron IDs: 0..n_excitatory-1 are excitatory (RS),
    n_excitatory..n_excitatory+n_inhibitory-1 are inhibitory (FS).

    Args:
        n_excitatory: Number of excitatory (RS) neurons.
        n_inhibitory: Number of inhibitory (FS) neurons.
        w_exc_to_inh: Weight of E->I synapses.
        w_inh_to_exc: Weight of I->E synapses (stored as negative).
        plasticity: Whether STDP is active.
        tau_syn: Post-synaptic current time constant (ms).

    Returns:
        A wired Network ready for simulation.
    """
    net = Network(plasticity=plasticity, tau_syn=tau_syn)

    for i in range(n_excitatory):
        net.add_neuron(i, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))

    for i in range(n_inhibitory):
        nid = n_excitatory + i
        net.add_neuron(nid, IzhikevichNeuron.from_type(NeuronType.FAST_SPIKING))

    # E -> I: every excitatory connects to every inhibitory
    for e_id in range(n_excitatory):
        for i in range(n_inhibitory):
            inh_id = n_excitatory + i
            net.connect(
                e_id,
                inh_id,
                Synapse(
                    weight=w_exc_to_inh,
                    synapse_type=SynapseType.EXCITATORY,
                ),
            )

    # I -> E: every inhibitory connects back to every excitatory
    for i in range(n_inhibitory):
        inh_id = n_excitatory + i
        for e_id in range(n_excitatory):
            net.connect(
                inh_id,
                e_id,
                Synapse(
                    weight=w_inh_to_exc,
                    synapse_type=SynapseType.INHIBITORY,
                ),
            )

    return net
