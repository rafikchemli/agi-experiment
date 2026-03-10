"""Layer builders for biologically plausible circuits.

Builds standard cortical circuit motifs:
- Retina -> V1 with STDP input synapses and lateral inhibition (WTA)

Neuron ID layout for V1 circuit:
- [0, n_pixels): input (retinal) neurons
- [n_pixels, n_pixels + n_v1_exc): V1 excitatory (RS)
- [n_pixels + n_v1_exc, ...): V1 inhibitory (FS)
"""

import numpy as np

from brain_sim.network import Network
from brain_sim.neuron import IzhikevichNeuron, NeuronType
from brain_sim.synapse import STDPParams, Synapse, SynapseType


def build_v1_circuit(
    grid_size: int = 8,
    n_v1_excitatory: int = 4,
    n_v1_inhibitory: int = 1,
    w_input_to_v1: float = 3.0,
    w_v1_to_inh: float = 5.0,
    w_inh_to_v1: float = 15.0,
    inh_tonic: float = 3.0,
    tau_syn: float = 5.0,
    stdp_params: STDPParams | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Network, dict[str, list[int]]]:
    """Build a retina -> V1 circuit with WTA lateral inhibition.

    Input neurons are driven by external currents (rate coding from images).
    V1 excitatory neurons receive input through STDP plastic synapses.
    V1 inhibitory neurons provide lateral inhibition (WTA) with fixed synapses.

    Initial input->V1 weights are randomized to break symmetry, which is
    essential for different V1 neurons to specialize for different features.

    Args:
        grid_size: Input image size (grid_size x grid_size pixels).
        n_v1_excitatory: Number of V1 excitatory neurons.
        n_v1_inhibitory: Number of V1 inhibitory neurons.
        w_input_to_v1: Mean initial weight for input->V1 synapses.
        w_v1_to_inh: Weight for V1->inhibitory synapses (fixed).
        w_inh_to_v1: Weight for inhibitory->V1 synapses (fixed).
        inh_tonic: Tonic current for inhibitory neurons.
        tau_syn: PSC decay time constant (ms).
        stdp_params: STDP parameters for input->V1 synapses.
        rng: NumPy random generator for reproducible weight initialization.

    Returns:
        Tuple of (network, id_map) where id_map has keys:
        "input", "v1_exc", "v1_inh" mapping to lists of neuron IDs.
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
    net = Network(plasticity=True, tau_syn=tau_syn)

    # Input (retinal) neurons
    input_ids = list(range(n_pixels))
    for nid in input_ids:
        net.add_neuron(nid, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))

    # V1 excitatory neurons
    v1_start = n_pixels
    v1_exc_ids = list(range(v1_start, v1_start + n_v1_excitatory))
    for nid in v1_exc_ids:
        net.add_neuron(nid, IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING))

    # V1 inhibitory neurons
    inh_start = v1_start + n_v1_excitatory
    v1_inh_ids = list(range(inh_start, inh_start + n_v1_inhibitory))
    for nid in v1_inh_ids:
        net.add_neuron(nid, IzhikevichNeuron.from_type(NeuronType.FAST_SPIKING))

    # Input -> V1: all-to-all, excitatory, STDP plastic, random initial weights
    for inp_id in input_ids:
        for v1_id in v1_exc_ids:
            w = float(rng.normal(w_input_to_v1, w_input_to_v1 * 0.3))
            w = max(stdp_params.w_min, min(stdp_params.w_max, w))
            net.connect(
                inp_id,
                v1_id,
                Synapse(
                    weight=w,
                    synapse_type=SynapseType.EXCITATORY,
                    stdp_params=stdp_params,
                    plastic=True,
                ),
            )

    # V1 excitatory -> V1 inhibitory: fixed (no STDP)
    for v1_id in v1_exc_ids:
        for inh_id in v1_inh_ids:
            net.connect(
                v1_id,
                inh_id,
                Synapse(
                    weight=w_v1_to_inh,
                    synapse_type=SynapseType.EXCITATORY,
                    plastic=False,
                ),
            )

    # V1 inhibitory -> V1 excitatory: fixed (no STDP)
    for inh_id in v1_inh_ids:
        for v1_id in v1_exc_ids:
            net.connect(
                inh_id,
                v1_id,
                Synapse(
                    weight=w_inh_to_v1,
                    synapse_type=SynapseType.INHIBITORY,
                    plastic=False,
                ),
            )

    id_map = {
        "input": input_ids,
        "v1_exc": v1_exc_ids,
        "v1_inh": v1_inh_ids,
    }
    return net, id_map


def get_receptive_fields(
    net: Network,
    v1_exc_ids: list[int],
    n_inputs: int,
) -> dict[int, np.ndarray]:
    """Extract receptive fields (input weight vectors) for V1 neurons.

    Each V1 neuron's receptive field is the vector of weights from all
    input neurons. When reshaped to the image grid, it shows what spatial
    pattern the neuron has learned to detect.

    Args:
        net: The network containing the V1 circuit.
        v1_exc_ids: IDs of V1 excitatory neurons.
        n_inputs: Number of input neurons.

    Returns:
        Dict of v1_id -> weight vector (1D array of length n_inputs).
    """
    fields: dict[int, np.ndarray] = {}
    for v1_id in v1_exc_ids:
        weights = np.zeros(n_inputs, dtype=np.float64)
        for conn in net.incoming_connections(v1_id):
            if conn.synapse.plastic and conn.pre_id < n_inputs:
                weights[conn.pre_id] = conn.synapse.weight
        fields[v1_id] = weights
    return fields
