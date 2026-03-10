"""Shared test fixtures for brain_sim tests."""

import pytest

from brain_sim.neuron import IzhikevichNeuron, NeuronType


@pytest.fixture
def rs_neuron() -> IzhikevichNeuron:
    """Regular Spiking excitatory neuron."""
    return IzhikevichNeuron.from_type(NeuronType.REGULAR_SPIKING)


@pytest.fixture
def fs_neuron() -> IzhikevichNeuron:
    """Fast Spiking inhibitory interneuron."""
    return IzhikevichNeuron.from_type(NeuronType.FAST_SPIKING)


@pytest.fixture
def ib_neuron() -> IzhikevichNeuron:
    """Intrinsically Bursting neuron."""
    return IzhikevichNeuron.from_type(NeuronType.INTRINSICALLY_BURSTING)
