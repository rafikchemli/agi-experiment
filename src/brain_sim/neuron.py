"""Izhikevich neuron model.

Implements the 2003 Izhikevich model — a 2-variable ODE that reproduces
20+ known cortical firing patterns with only 4 parameters (a, b, c, d).

These 4 parameters are the "genetic code" of each neuron type. They are
set at initialization and not learned.

Reference: Izhikevich, E.M. (2003). Simple model of spiking neurons.
IEEE Transactions on Neural Networks, 14(6), 1569-1572.
"""

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field


class NeuronType(StrEnum):
    """Biologically identified neuron types with known Izhikevich parameters.

    Each type maps to a specific cortical cell class observed in experiments.
    """

    REGULAR_SPIKING = "regular_spiking"
    FAST_SPIKING = "fast_spiking"
    INTRINSICALLY_BURSTING = "intrinsically_bursting"


class NeuronParams(BaseModel):
    """Izhikevich neuron parameters — the 'genetic code'.

    Args:
        a: Time scale of recovery variable u. Smaller = slower recovery.
        b: Sensitivity of u to subthreshold fluctuations of v.
        c: After-spike reset value of membrane potential v (mV).
        d: After-spike reset increment of recovery variable u.
    """

    a: float = Field(description="Recovery time scale")
    b: float = Field(description="Recovery sensitivity to v")
    c: float = Field(description="Post-spike reset voltage (mV)")
    d: float = Field(description="Post-spike recovery increment")


# Genetic presets — these are the known parameter sets from Izhikevich 2003
NEURON_PRESETS: dict[NeuronType, NeuronParams] = {
    NeuronType.REGULAR_SPIKING: NeuronParams(a=0.02, b=0.2, c=-65.0, d=8.0),
    NeuronType.FAST_SPIKING: NeuronParams(a=0.1, b=0.2, c=-65.0, d=2.0),
    NeuronType.INTRINSICALLY_BURSTING: NeuronParams(a=0.02, b=0.2, c=-55.0, d=4.0),
}


class IzhikevichNeuron:
    """A single Izhikevich spiking neuron.

    Implements the 2-variable model:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        if v >= 30 mV: v = c, u = u + d

    The neuron maintains internal state (v, u) that persists between timesteps.
    This is a dynamical system, not a static function.

    Args:
        params: Izhikevich parameters (a, b, c, d).
        neuron_type: Optional label for the biological type.
    """

    SPIKE_THRESHOLD: float = 30.0  # mV — standard Izhikevich threshold
    RESTING_V: float = -65.0  # mV — resting membrane potential
    RESTING_U_FACTOR: float = -14.0  # u_rest = b * v_rest (for b=0.2, v=-65 -> u=-13)

    def __init__(
        self,
        params: NeuronParams,
        neuron_type: NeuronType | None = None,
    ) -> None:
        self.params = params
        self.neuron_type = neuron_type

        # Internal state — this persists, the neuron is always "alive"
        self.v: float = self.RESTING_V  # membrane potential (mV)
        self.u: float = params.b * self.RESTING_V  # recovery variable

        # Spike tracking
        self.fired: bool = False  # did the neuron spike this timestep?
        self.spike_times: list[int] = []  # all spike times (ms)
        self._timestep: int = 0

    @classmethod
    def from_type(cls, neuron_type: NeuronType) -> "IzhikevichNeuron":
        """Create a neuron from a biological type preset.

        Args:
            neuron_type: The biological neuron type.

        Returns:
            A neuron initialized with the preset parameters.
        """
        return cls(params=NEURON_PRESETS[neuron_type], neuron_type=neuron_type)

    def step(self, current: float, dt: float = 1.0) -> bool:
        """Advance the neuron by one timestep.

        Uses the Izhikevich ODE with 0.5ms sub-steps for numerical stability
        (as recommended in the original paper for dt=1ms).

        Args:
            current: Total input current (from synapses + external).
            dt: Timestep in ms. Default 1.0ms.

        Returns:
            True if the neuron spiked this timestep.
        """
        self.fired = False
        a, b, c, d = self.params.a, self.params.b, self.params.c, self.params.d
        v, u = self.v, self.u

        # Two 0.5ms sub-steps for numerical stability (Izhikevich recommendation)
        half_dt = dt / 2.0
        for _ in range(2):
            if v >= self.SPIKE_THRESHOLD:
                break
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + current) * half_dt
            v += dv
            du = a * (b * v - u) * half_dt
            u += du

        # Spike detection and reset
        if v >= self.SPIKE_THRESHOLD:
            v = c
            u = u + d
            self.fired = True
            self.spike_times.append(self._timestep)

        self.v = v
        self.u = u
        self._timestep += 1
        return self.fired

    def reset(self) -> None:
        """Reset the neuron to resting state.

        Used for starting new experiments, not during normal operation.
        """
        self.v = self.RESTING_V
        self.u = self.params.b * self.RESTING_V
        self.fired = False
        self.spike_times = []
        self._timestep = 0

    @property
    def firing_rate(self) -> float:
        """Compute firing rate in Hz from spike history.

        Returns:
            Firing rate in spikes per second (Hz). Returns 0 if no time elapsed.
        """
        if self._timestep == 0:
            return 0.0
        duration_s = self._timestep / 1000.0  # ms to seconds
        return len(self.spike_times) / duration_s


def simulate_neuron(
    neuron: IzhikevichNeuron,
    current: float,
    duration_ms: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Run a neuron for a given duration with constant input current.

    Args:
        neuron: The neuron to simulate.
        current: Constant input current.
        duration_ms: Simulation duration in milliseconds.

    Returns:
        Tuple of (voltage_trace, recovery_trace, spike_times).
    """
    neuron.reset()
    v_trace = np.zeros(duration_ms, dtype=np.float64)
    u_trace = np.zeros(duration_ms, dtype=np.float64)

    for t in range(duration_ms):
        neuron.step(current)
        v_trace[t] = neuron.v
        u_trace[t] = neuron.u

    return v_trace, u_trace, list(neuron.spike_times)
