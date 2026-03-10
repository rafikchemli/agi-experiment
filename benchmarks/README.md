# Benchmarks: Can Local Learning Beat Backprop on MNIST?

The brain learns without backpropagation. Every synapse update uses only information available at that synapse — presynaptic activity, postsynaptic activity, and diffuse neuromodulatory signals. No global error signal. No weight transport. No gradient computation.

This benchmark suite asks: **how close can biologically plausible, local-only learning rules get to a standard backprop MLP on MNIST?**

## Results

| Approach | Accuracy | Time | Backprop? | Learning Rule |
|----------|----------|------|-----------|---------------|
| Backprop MLP (baseline) | 97.3% | 7s | YES | SGD + cross-entropy |
| Forward-Forward (Hinton 2022) | 96.9% | 334s | NO | Local contrastive goodness |
| Sparse Predictive Coding | 89.1% | 522s | NO | Hebbian + ISTA settling |

The backprop MLP is the ceiling — a well-tuned control group. Everything else uses only local learning rules with biological analogues.

## Approaches

### Backprop MLP (`backprop_mlp.py`)

The control group. Standard feedforward network: 784 → 300 (ReLU) → 10 (softmax), trained with SGD and cross-entropy loss. Step LR decay (halve every 3 epochs), 15 epochs. This proves the task is solvable and sets the bar.

### Forward-Forward (`forward_forward.py`)

Hinton's 2022 algorithm. Each layer learns independently by maximizing "goodness" (sum of squared activations) for correct inputs and minimizing it for incorrect ones. No gradient flows between layers.

- **Architecture**: 784 → 500 (ReLU) → 500 (ReLU)
- **Learning**: Per-layer contrastive loss with local Adam optimizer
- **Inference**: Try all 10 labels, pick highest total goodness across layers
- **Bio analogue**: Local Hebbian-like updates, no inter-layer error signal

### Sparse Predictive Coding (`predictive_coding.py`)

The brain reconstructs, then classifies. V1 neurons learn a dictionary of visual features via sparse coding (Olshausen & Field 1996). Each image is encoded as a sparse combination of these features. Classification is a downstream readout.

- **Architecture**: Dictionary D (784 × 500) + Readout W (500 × 10)
- **Inference**: ISTA settling (75 iterations) — feedforward drive competes with lateral inhibition until a sparse code emerges
- **Dictionary learning**: ΔD ∝ residual × codes (Hebbian: error × presynaptic activity)
- **Readout learning**: ΔW ∝ codes × label_error (reward-modulated Hebbian)
- **Uncertainty**: Reconstruction error — high error means "I can't explain this input"

See [architecture_exploration.md](approaches/architecture_exploration.md) for the full exploration log, including what failed (v1: 11.3%) and what worked (v2: 89.1%).

## Architecture Variants Under Exploration

From the [exploration plan](approaches/architecture_exploration.md):

- **FISTA (momentum)** — accelerated ISTA, faster convergence with fewer settling steps
- **Two-phase training** — unsupervised dictionary, then supervised readout (more biological)
- **Hierarchical sparse coding** — two dictionary layers (V1 → V2)
- **Energy-based classification** — try all 10 class hypotheses, pick lowest reconstruction error
- **Precision-weighted prediction error** — learn per-feature uncertainty (attention mechanism)

## Running

```bash
# Run all approaches
uv run python -m benchmarks.evaluate

# Run a single approach
uv run python -m benchmarks.evaluate predictive_coding

# Re-plot saved results without retraining
uv run python -m benchmarks.evaluate --plot

# Hyperparameter sweep (Optuna) for predictive coding
uv run python -m benchmarks.sweep --n-trials 50
```

Results accumulate in `results/benchmark_results.json` — re-running an approach updates its entry without losing others.

## Adding a New Approach

Drop a `.py` file in `approaches/` with a class that inherits from `MNISTApproach`:

```python
from benchmarks.base import EpochMetrics, MNISTApproach

class MyApproach(MNISTApproach):
    name = "my_approach"
    uses_backprop = False

    def train(self, images, labels):
        # ... your learning rule ...
        self.history.append(EpochMetrics(epoch=1, train_acc=0.9, loss=0.1))

    def predict(self, images):
        # ... your inference ...
        return predictions
```

It will be auto-discovered and appear in the benchmark table and training curves plot. No registration needed.

## Project Structure

```
benchmarks/
├── base.py                         # MNISTApproach ABC + EpochMetrics
├── mnist_loader.py                 # Download, cache, and serve MNIST
├── evaluate.py                     # Train, evaluate, compare, plot
├── sweep.py                        # Optuna hyperparameter search
├── approaches/
│   ├── architecture_exploration.md # Exploration log and variant ideas
│   ├── backprop_mlp.py             # Control group (97.3%)
│   ├── forward_forward.py          # Hinton 2022 (96.9%)
│   ├── predictive_coding.py        # Sparse coding + Hebbian (89.1%)
│   └── sparse_coding_v2_89pct.py   # Historical v2 snapshot
└── results/
    ├── benchmark_results.json      # Accumulated results (all approaches)
    └── benchmark_comparison.png    # Two-panel plot (bars + training curves)
```

## The Goal

Find a learning algorithm that:
1. Uses **no backpropagation** — all learning rules are local
2. Achieves **>97% on MNIST** — matching the backprop baseline
3. Every operation has a **biological analogue** in real neural circuits
4. Includes **uncertainty estimation** for free

Forward-Forward is already at 96.9%. Sparse predictive coding is at 89.1% and climbing. The gap is closing.
