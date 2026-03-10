"""Homeostatic plasticity and learning rules.

Homeostatic synaptic scaling (Turrigiano et al., 1998):
Neurons regulate their own firing rate by multiplicatively scaling
all incoming excitatory synaptic weights. This prevents runaway
excitation or silencing that STDP alone would cause.

In biology, homeostasis is SLOW (hours to days), operating as a
stabilizing complement to the FAST STDP rule (seconds to minutes).
In our simulation, we accelerate it for tractability but maintain
the principle: homeostasis corrects rate, STDP shapes selectivity.

Reference: Turrigiano, G.G. et al. (1998). Activity-dependent scaling
of quantal amplitude in neocortical neurons. Nature, 391, 892-896.
"""

from brain_sim.network import Network
from brain_sim.synapse import SynapseType


def apply_homeostatic_scaling(
    net: Network,
    target_rates: dict[int, float],
    window_ms: int = 1000,
    eta: float = 0.01,
) -> dict[int, float]:
    """Scale incoming excitatory weights to maintain target firing rates.

    For each neuron in target_rates:
    1. Measure actual firing rate over the recent window
    2. Compute scaling factor proportional to rate error
    3. Multiplicatively scale all incoming excitatory synapse weights

    Only excitatory incoming synapses are scaled — inhibitory synapses
    are left untouched (consistent with biology: homeostasis acts on
    excitatory AMPA/NMDA receptors, not inhibitory GABA receptors).

    Args:
        net: The network to apply homeostasis to.
        target_rates: Target firing rate (Hz) per neuron ID.
        window_ms: Measurement window for computing actual rate (ms).
        eta: Scaling strength. Larger = more aggressive correction.

    Returns:
        Dict of neuron_id -> scaling factor applied.
    """
    scaling_factors: dict[int, float] = {}

    for nid, target_hz in target_rates.items():
        recent = net.recent_spike_count(nid, window_ms)
        actual_hz = recent / (window_ms / 1000.0)

        if target_hz > 0 and actual_hz > 0:
            factor = 1.0 + eta * (target_hz - actual_hz) / target_hz
        elif actual_hz == 0:
            factor = 1.0 + eta
        else:
            factor = 1.0

        # Bound scaling to prevent catastrophic single-step changes
        factor = max(0.8, min(1.2, factor))

        # Scale incoming excitatory weights only
        for conn in net.incoming_connections(nid):
            if conn.synapse.synapse_type == SynapseType.EXCITATORY:
                conn.synapse.weight = max(
                    conn.synapse.stdp.w_min,
                    min(conn.synapse.stdp.w_max, conn.synapse.weight * factor),
                )

        scaling_factors[nid] = factor

    return scaling_factors
