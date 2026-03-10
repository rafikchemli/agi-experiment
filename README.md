# agi_experiment

Biologically plausible spiking neural network simulator built from first principles.
Pure Python + NumPy. No PyTorch, no JAX, no backpropagation.

Three research tracks exploring whether local learning rules can produce intelligent behavior:

1. **Core Simulator** — Izhikevich neurons, STDP synapses, winner-take-all circuits, homeostatic plasticity
2. **Benchmarks** — Can local learning match backprop on MNIST? (Answer: DFA hits **97.5%**, beating backprop's 97.3%)
3. **Causal Dictionaries** — Do composable causal primitives emerge from unsupervised learning? (Answer: **yes, 5/5 composition tests pass**)

## Quick Start

```bash
git clone https://github.com/yourname/agi_experiment.git
cd agi_experiment
make init    # installs uv, syncs deps, sets up pre-commit
make check   # format + lint + typecheck + test
```

## Core Simulator

Seven modules in `src/brain_sim/` implementing a biologically faithful spiking neural network:

| Module | What it does | Biology |
|--------|-------------|---------|
| `neuron.py` | Izhikevich 2-variable ODE model | Regular spiking, fast spiking, intrinsically bursting neurons |
| `synapse.py` | STDP plasticity with eligibility traces | Bi & Poo 1998 timing rule, Dale's Law, conduction delays |
| `network.py` | Clock-driven simulation (per-object) | 5-phase biological step: decay, deliver, fire, propagate, trace |
| `spike_net.py` | Vectorized numpy implementation (~100x faster) | Same biology, BLAS matrix operations |
| `layers.py` | V1 circuit builder (retina to primary visual cortex) | Receptive field emergence via STDP + WTA |
| `encoding.py` | Image to firing current (rate coding) | Retinal ganglion cell preprocessing |
| `learning.py` | Homeostatic synaptic scaling | Turrigiano et al. 1998, prevents STDP runaway |

The self-organization test (`tests/test_network.py`) verifies that the V1 circuit actually learns orientation selectivity from scratch — different neurons specialize for horizontal vs. vertical bars through STDP alone.

## Benchmark Results

Can local learning rules match backpropagation on MNIST?

| Approach | Accuracy | Backprop? | Key idea |
|----------|----------|-----------|----------|
| **DFA v20** | **97.5%** | **No** | Direct Feedback Alignment (Lillicrap et al. 2016) |
| Backprop MLP | 97.3% | Yes | Standard 784-300-10, SGD + cross-entropy |
| Forward-Forward | 96.9% | No | Hinton 2022, layer-local contrastive goodness |
| Augmented SC v9 | 96.7% | No | Sparse coding + microsaccade augmentation |
| Energy SC v6 | 96.4% | No | Dictionary incoherence regularization |
| Predictive Coding | 92.7% | No | ISTA + Hebbian dictionary + reward readout |

20+ approaches explored in `benchmarks/approaches/`, with full research log in `benchmarks/approaches/architecture_exploration.md`.

```bash
# Run a specific benchmark approach
make benchmark APPROACH=dfa_v20
```

## Causal Dictionary Learning

Can a sparse dictionary discover composable causal rules from raw physical events?

A micro-world simulates three physics rules (gravity, containment, contact). A dictionary learns atoms from individual rule events, then is tested on novel compositions it has never seen.

| Architecture | Mean Pass | Best | Worst | Key insight |
|-------------|-----------|------|-------|-------------|
| **ProductOfExperts** (3+3) | **4.4/5** | **5/5** | 2/5 | Factored rule/position codebooks |
| **ContrastiveDictionary** | **4.4/5** | **5/5** | 4/5 | ISTA + contrastive specialization pressure |
| SlotDictionary | 4.0/5 | 4/5 | 4/5 | Competitive slot binding |
| ISTA baseline | 3.8/5 | 5/5 | 3/5 | Standard sparse coding |

The breakthrough: separating *what rule* from *where it happens* (ProductOfExperts) lets gravity get a compact 3-atom rule code regardless of positional diversity.

```bash
# Run the causal dictionary experiment
make experiment

# Try different architectures
make experiment ARGS="--arch contrastive"
make experiment ARGS="--arch ista --n-atoms 8"
```

Full experiment log: `experiments/causal_dictionaries/results/lab_notebook.md`

## Project Structure

```
agi_experiment/
├── src/brain_sim/                          # Core spiking neural network simulator
│   ├── neuron.py                           #   Izhikevich neuron model
│   ├── synapse.py                          #   STDP synapses with eligibility traces
│   ├── network.py                          #   Per-object network simulation
│   ├── spike_net.py                        #   Vectorized numpy implementation
│   ├── layers.py                           #   V1 circuit builder
│   ├── encoding.py                         #   Image-to-current rate coding
│   └── learning.py                         #   Homeostatic synaptic scaling
├── experiments/causal_dictionaries/        # Causal dictionary learning experiment
│   ├── micro_world.py                      #   5x5 grid world physics engine
│   ├── event_encoding.py                   #   Event-to-vector encoding (3 schemes)
│   ├── sparse_dictionary.py                #   ISTA sparse coding
│   ├── architectures.py                    #   ProductOfExperts, Slot, Contrastive
│   ├── analysis.py                         #   Specialization + composition metrics
│   ├── run.py                              #   End-to-end POC runner
│   └── results/                            #   Results, visualizations, lab notebook
├── benchmarks/                             # MNIST benchmark framework
│   ├── base.py                             #   MNISTApproach abstract base class
│   ├── mnist_loader.py                     #   MNIST download and serve
│   ├── evaluate.py                         #   Train, evaluate, compare approaches
│   └── approaches/                         #   20+ learning algorithm implementations
├── tests/                                  # pytest test suite
│   ├── test_neuron.py                      #   Izhikevich dynamics
│   ├── test_synapse.py                     #   STDP timing, Dale's Law, eligibility
│   ├── test_learning.py                    #   WTA, homeostasis, self-organization
│   ├── test_network.py                     #   Encoding, V1 circuit, emergence
│   ├── test_spike_net.py                   #   Vectorized population/projection
│   ├── test_causal_dictionaries.py         #   Micro-world, encoding, dictionary
│   └── test_forward_forward.py             #   Forward-Forward algorithm
├── Makefile                                # make init / check / test / benchmark
├── pyproject.toml                          # Python 3.12, uv, ruff, mypy strict
└── LICENSE                                 # MIT
```

## Development

```bash
make init       # setup (installs uv, syncs deps, pre-commit hooks)
make check      # format + lint + typecheck + test
make test       # just tests
make format     # just formatting
make lint       # just linting
make typecheck  # just mypy
make clean      # remove build artifacts
```

**Stack**: Python 3.12 | uv | ruff | mypy (strict) | pytest

## References

- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.
- Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*, 18(24), 10464-10472.
- Turrigiano, G. G. et al. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, 391, 892-896.
- Lillicrap, T. P. et al. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*, 7, 13276.
- Hinton, G. (2022). The Forward-Forward Algorithm. *arXiv:2212.13345*.
- Olshausen, B. A. & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381, 607-609.

## License

MIT
