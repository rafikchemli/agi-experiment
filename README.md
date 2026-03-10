# agi_experiment

**Can machines discover the rules of physics by watching objects move?**

This project builds a tiny physical world — objects on a grid governed by gravity, containment, and contact — then asks whether unsupervised sparse dictionary learning can discover the underlying causal rules from raw observations alone. The key test: can the learned representations generalize to *novel compositions* of rules the system has never seen?

**The answer is yes.** A ProductOfExperts architecture that separates *what rule* from *where it happens* passes **5/5 composition tests** — correctly reconstructing events involving multiple simultaneous physics rules from single-rule atoms.

## The Experiment

### The Micro-World

A 5x5 grid world with named objects (ball, square, triangle) and three physics rules:

- **Gravity**: unsupported objects fall to the nearest surface
- **Containment**: objects inside a container move when the container moves
- **Contact/Push**: objects pushed into neighbors transfer momentum

The world generates thousands of events — each a (rule, objects, positions, outcome) tuple describing one causal interaction.

### What We Learn

Events are encoded into 19-dimensional continuous vectors and fed to a sparse dictionary. The dictionary learns a small set of **atoms** — basis vectors that can be combined to reconstruct any event. The question: do these atoms correspond to meaningful causal primitives?

### The Composition Test

The critical test: generate events where **multiple rules act simultaneously** (e.g., gravity + containment: a ball falls inside a moving square). The dictionary has only seen single-rule events during training. Can it reconstruct multi-rule events by *composing* the atoms it learned?

| Test | Rules Combined | Pass Criteria |
|------|---------------|---------------|
| T1 | gravity + containment | reconstruction ratio < 2.0, Jaccard > 0.7 |
| T2 | gravity + contact | reconstruction ratio < 2.0, Jaccard > 0.7 |
| T3 | containment + contact | reconstruction ratio < 2.0, Jaccard > 0.7 |
| T4 | all three | reconstruction ratio < 2.0, Jaccard > 0.7 |
| T5 | all three (variant) | reconstruction ratio < 2.0, Jaccard > 0.7 |

### Results

| Architecture | Mean Pass (5 seeds) | Best | Key Insight |
|-------------|-----------|------|-------------|
| **ProductOfExperts** (3+3) | **4.4/5** | **5/5** | Factored rule/position codebooks |
| **ContrastiveDictionary** | **4.4/5** | **5/5** | ISTA + contrastive specialization pressure |
| SlotDictionary | 4.0/5 | 4/5 | Competitive slot binding |
| ISTA baseline | 3.8/5 | 5/5 | Standard sparse coding |

The breakthrough: the ProductOfExperts architecture maintains separate codebooks for rule identity (3 atoms) and spatial position (3 atoms). This lets gravity get a compact code regardless of positional diversity — the factored representation composes naturally because the factors are independent.

### Run It

```bash
git clone https://github.com/rafikchemli/agi-experiment.git
cd agi-experiment
make init       # installs uv, syncs deps

# Run the full experiment (default: ProductOfExperts)
make experiment

# Try different architectures
make experiment ARGS="--arch contrastive"
make experiment ARGS="--arch ista --n-atoms 8"
make experiment ARGS="--arch product-of-experts --seed 7"
```

Full experiment log with all 8 experiments: [experiments/causal_dictionaries/results/lab_notebook.md](experiments/causal_dictionaries/results/lab_notebook.md)

## What Else Is Here

The causal dictionary experiment sits on top of a broader research platform:

### Core Spiking Simulator (`src/brain_sim/`)

A biologically plausible spiking neural network built from first principles — Izhikevich neurons, STDP synapses, winner-take-all circuits, homeostatic plasticity. The V1 circuit test verifies that orientation-selective neurons emerge purely from local learning rules.

### MNIST Benchmarks (`benchmarks/`)

Can local learning rules match backpropagation? 20+ approaches explored, with Direct Feedback Alignment reaching **97.5%** (beating backprop's 97.3%).

| Approach | Accuracy | Backprop? |
|----------|----------|-----------|
| **DFA v20** | **97.5%** | No |
| Backprop MLP | 97.3% | Yes |
| Forward-Forward | 96.9% | No |
| Sparse Coding v9 | 96.7% | No |

```bash
make benchmark APPROACH=dfa_v20
```

## Project Structure

```
agi_experiment/
├── experiments/causal_dictionaries/        # Main experiment
│   ├── micro_world.py                      #   5x5 grid physics engine
│   ├── event_encoding.py                   #   Event → 19-dim vector (3 schemes)
│   ├── sparse_dictionary.py                #   ISTA sparse coding
│   ├── architectures.py                    #   ProductOfExperts, Slot, Contrastive
│   ├── analysis.py                         #   Specialization + composition metrics
│   ├── run.py                              #   End-to-end runner
│   └── results/                            #   Results, plots, lab notebook
├── src/brain_sim/                          # Spiking neural network simulator
│   ├── neuron.py                           #   Izhikevich neuron model
│   ├── synapse.py                          #   STDP with eligibility traces
│   ├── network.py                          #   Per-object simulation
│   ├── spike_net.py                        #   Vectorized numpy (~100x faster)
│   ├── layers.py                           #   V1 circuit builder
│   ├── encoding.py                         #   Rate coding
│   └── learning.py                         #   Homeostatic scaling
├── benchmarks/                             # MNIST benchmark framework
│   ├── approaches/                         #   20+ learning algorithms
│   └── results/                            #   Comparison data
├── tests/                                  # 149 tests, all passing
├── Makefile                                # make init / check / experiment / benchmark
└── pyproject.toml                          # Python 3.12, uv, ruff, mypy strict
```

## Development

```bash
make init       # setup
make check      # format + lint + typecheck + test (149 tests)
make test       # just tests
make clean      # remove build artifacts
```

**Stack**: Python 3.12 | uv | ruff | mypy (strict) | pytest | NumPy

## References

- Olshausen, B. A. & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381, 607-609.
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.
- Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*, 18(24), 10464-10472.
- Turrigiano, G. G. et al. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*, 391, 892-896.
- Lillicrap, T. P. et al. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*, 7, 13276.
- Hinton, G. (2022). The Forward-Forward Algorithm. *arXiv:2212.13345*.

## License

MIT
