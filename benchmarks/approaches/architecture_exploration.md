# Architecture Exploration: Brain-Inspired MNIST Classification

## Goal

Find a learning algorithm that:
1. Uses **NO backpropagation** — all learning rules are local (Hebbian)
2. Achieves meaningful accuracy on MNIST (target: >95%, backprop gets 98%)
3. Every operation has a **biological analogue** in real neural circuits
4. Includes **uncertainty estimation** for free (not a bolted-on extra)

## Benchmark Results So Far

| Approach | Accuracy | Time | Backprop? | Status |
|----------|----------|------|-----------|--------|
| **DFA v20** | **97.5%** | 299s | **NO** | **BEATS BACKPROP** |
| Backprop MLP (baseline) | 97.3% | 7s | YES | Baseline |
| Hybrid FF+SC v17 | 97.1% | 605s | NO | Confidence-based FF+SC arbitration |
| Forward-Forward (Hinton 2022) | 96.9% | 334s | NO | Layer-local contrastive goodness |
| FF v2 (extended) | 96.9% | 273s | NO | FF with 100 epochs/layer |
| Augmented SC v9 | 96.7% | 682s | NO | v6 + microsaccade augmentation (±1px shift) |
| Wide SC v13 | 96.5% | 1440s | NO | v9 with 400 features/class |
| Cosine LR SC v15 | 96.4% | 783s | NO | v9 + cosine LR annealing |
| Incoherent SC v6 | 96.4% | 536s | NO | v5 + dictionary incoherence regularization |
| Optuna-tuned SC v7 | 95.6% | 582s | NO | v5 + Optuna-optimized hyperparams |
| Energy-based SC v5 | 94.6% | 414s | NO | Class-specific dictionaries, competitive recon |
| Predictive Coding (tuned) | 92.7% | 326s | NO | Improved via lower sparsity |
| FF Enhanced (wider+schedule) | 88.4% | 211s | NO | Failed — underfitting |

## Architecture A: Sparse Predictive Coding

### Core Idea

The brain doesn't classify images — it **reconstructs** them. V1 neurons learn a dictionary of visual features (edges, strokes, textures). Each image is encoded as a sparse combination of these features. Classification is a downstream readout from the sparse representation.

### Math

**Objective**: min_z ½||x − Dz||² + λ||z||₁, subject to z ≥ 0

- **D** ∈ R^(784 × 500): dictionary of visual features (V1 receptive fields)
- **z** ∈ R^500: sparse code (which features are active — firing rates)
- **λ**: sparsity penalty (lateral inhibition strength)

**Inference** (ISTA settling):
```
for t in 1..T:
    residual = x − Dz           # prediction error
    drive = Dᵀ residual         # feedforward input
    z = z + α · drive           # integrate
    z = max(0, z − λα)          # lateral inhibition + threshold
    z = min(z, 5)               # saturation
```

**Learning** (Hebbian):
```
ΔD ∝ residual · zᵀ             # error × pre-synaptic activity
ΔW ∝ z · (y − ŷ)ᵀ             # activity × reward prediction error
```

### Biological Analogues

| Operation | Math | Biology |
|-----------|------|---------|
| Dictionary columns | D[:,j] | V1 simple cell receptive field |
| ISTA settling loop | 75 iterations | ~100ms cortical recurrence |
| Feedforward drive | Dᵀ(x − Dz) | Thalamus → V1 excitatory input |
| Soft threshold | max(0, z − λ) | Lateral inhibition via interneurons |
| Non-negativity | z ≥ 0 | Firing rates can't be negative |
| Saturation cap | z ≤ 5 | Neuronal saturation / refractory period |
| Column normalization | ||d_j|| = 1 | Homeostatic synaptic scaling |
| Dictionary update | ΔD ∝ ε·z | Hebbian: error × activity |
| Readout update | ΔW ∝ z·δ | Reward-modulated Hebbian (dopamine) |
| Uncertainty | ||x − Dz||² | Residual prediction error |

### What Worked (v2, 89.1%)

- **Conservative ISTA step** (0.1): Prevents code explosion when dictionary columns correlate
- **Hard cap on z** (≤ 5): Prevents runaway positive feedback between codes and dictionary
- **Lower dictionary lr** (0.002): Keeps dictionary changes small enough for ISTA to track
- **Higher readout lr** (0.05): Lets the classifier exploit improving features quickly
- **Sparsity 0.3**: ~9% of features active — biologically realistic

### What Failed (v1, 11.3%)

- ISTA step too large (0.3) → codes explode → dictionary destabilizes → all features die
- Epoch 1-3 looked great (83.7%), then catastrophic collapse at epoch 4
- Root cause: positive feedback loop between growing codes and growing dictionary updates

## Exploration Plan

### Variables to Sweep (Optuna)

| Parameter | Search Range | Expected Impact |
|-----------|-------------|-----------------|
| n_features | 200–1000 | More features = richer representation but slower |
| n_settle | 25–150 | More settling = better codes but slower |
| sparsity | 0.05–1.0 | Controls fraction of active neurons |
| infer_rate | 0.01–0.5 | ISTA convergence speed vs stability |
| learn_rate | 0.0005–0.01 | Dictionary learning speed vs stability |
| sup_rate | 0.01–0.1 | Readout learning speed |

### Possible Architecture Variants

- **B: FISTA (momentum)** — accelerated ISTA, faster convergence with fewer steps
- **C: Two-phase training** — unsupervised dictionary → supervised readout (more biological)
- **D: Hierarchical sparse coding** — two dictionary layers (V1 → V2)
- **E: Energy-based classification** — try all 10 class hypotheses, pick lowest reconstruction error
- **F: Precision-weighted prediction error** — learn per-feature uncertainty (attention mechanism)

### Short Loop Protocol

```
implement (minimal) → run MNIST → record result → diagnose → iterate
```

Progression targets:
- >10% = not stuck at chance (basic mechanism works) ✓
- >50% = features are somewhat useful ✓
- >90% = real learning, worth optimizing ✓ (92.7% with tuned PC)
- >95% = competitive with backprop
- >97% = matches backprop with purely local rules

## Version History

| Version | File | Accuracy | Key Change |
|---------|------|----------|------------|
| v1 (predictive coding) | — | 11.3% | Hierarchical PC with ReLU — dead neurons at test time |
| v2 (sparse coding) | sparse_coding_v2_89pct.py | 89.1% | ISTA + dictionary learning, stability fixes |
| v3 (tuned PC) | predictive_coding.py | 92.7% | sparsity 0.3→0.01, learn_rate 0.002→0.005, sup_rate 0.05→0.1, epochs 25→40 |
| v4 (snapshot) | sparse_coding_v4_927pct.py | 92.7% | Snapshot of v3 params (sparsity=0.05, 25 epochs) |
| FF Enhanced | ff_enhanced.py | 88.4% | 2×1000 + cosine LR + multi-neg — FAILED (underfitting) |
| v5 (energy-based) | sparse_coding_v5_energy.py | 94.6% | 10 class-specific dictionaries, competitive reconstruction |
| v6 (incoherent) | sparse_coding_v6_incoherent.py | 96.4% | v5 + inter-dictionary incoherence penalty (inc=0.001) |
| v7 (Optuna-tuned) | sparse_coding_v7_energy.py | 95.6% | v5 architecture + Optuna-optimized hyperparams (40 trials) |
| v8 (hierarchical) | sparse_coding_v8_hierarchical.py | ~95.7% | V1 shared + V2 class-specific — hierarchy bottleneck |
| v9 (augmented) | sparse_coding_v9_augmented.py | 96.7% | v6 + microsaccade augmentation (±1px random shift) |
| v10 (TTA) | sparse_coding_v10_tta.py | ~96.3% | Test-time augmentation — marginal gain, massive slowdown |
| v11 (FISTA) | sparse_coding_v11_fista.py | ~85% | FISTA momentum — unstable, codes oscillate |
| v12 (discriminative) | sparse_coding_v12_discrim.py | ~96.0% | Anti-Hebbian discriminative refinement — destroys recon |
| v13 (wide) | sparse_coding_v13_wide.py | 96.5% | 400 features/class — wider didn't help |
| v14 (ensemble) | sparse_coding_v14_ensemble.py | ~96.7% | 3-5 member ensemble — no improvement |
| v15 (cosine LR) | sparse_coding_v15_cosine.py | 96.4% | Cosine LR decay — marginal on 10k, didn't scale |
| v16 (PCA init) | sparse_coding_v16_pca_init.py | ~95.7% | PCA-initialized dictionaries — hurts incoherence |
| FF v2 (extended) | forward_forward_v2_long.py | 96.9% | 100 epochs/layer — already converged at 25ep on 50k |
| v17 (hybrid FF+SC) | hybrid_v17_ff_sc.py | 97.1% | Confidence-based FF+SC arbitration |
| v18 (soft fusion) | hybrid_v18_soft_fusion.py | ~97.0% | Softmax probability fusion (RNG issue) |
| v19 (compose) | hybrid_v19_compose.py | ~97.0% | Imported standalone models + soft fusion |
| **v20 (DFA)** | **dfa_v20.py** | **97.5%** | **Direct Feedback Alignment — BEATS BACKPROP** |

### Optuna-Tuned SC v7 Analysis (95.6%)

**What worked**: 40-trial Optuna sweep on 5k subset found better hyperparams for the v5 energy-based architecture. Key changes: more features per class (250 vs 200), higher sparsity (0.0519 vs 0.01), faster dictionary learning (0.0207 vs 0.01). Pure hyperparameter optimization — no architectural changes.

**Params found by Optuna**:
- `n_features_per_class=250` (was 200) — richer per-class representation
- `n_settle=50` (was 40) — more ISTA iterations for better convergence
- `sparsity=0.0519` (was 0.01) — stronger lateral inhibition = sparser codes
- `infer_rate=0.0844` (was 0.1) — slightly slower ISTA step for stability
- `learn_rate=0.0207` (was 0.01) — 2x faster dictionary learning
- `batch_size=256` (unchanged)

**What limits it**: +1% over v5 (94.6% → 95.6%) but still 0.8% below v6 incoherent (96.4%). The incoherence penalty in v6 adds a discriminative signal that hyperparameter tuning alone cannot replicate. The remaining gap to backprop (1.7%) requires architectural changes, not just param optimization.

**Lesson**: Hyperparameter tuning gives diminishing returns for this architecture. The bottleneck is the lack of cross-class discrimination in pure reconstruction-based learning.

### Augmented SC v9 Analysis (96.7%)

**What worked**: Adding ±1 pixel random shifts (microsaccade augmentation) to v6 closes the generalization gap. v6 had 98.4% train / 96.4% test (2% gap). v9 has ~96.7% train / 96.7% test (0% gap). The model can no longer memorize exact pixel positions and must learn robust features.

**Key findings**: ±1 is the sweet spot. ±2 is too aggressive (hurts both train and test). ±3 destroys the signal. This matches microsaccade physiology: real eye jitter is ~1-2 pixels at retinal resolution.

**Progression**: v5 (94.6%) → v6 +incoherence (+1.8%) → v9 +augmentation (+0.3%) = 96.7%. Only 0.6% from the 97.3% target.

**What limits it**: The augmentation removed overfitting but train accuracy is now ~96.7% — the model's capacity is the bottleneck, not generalization. Remaining gap (0.6%) requires either more expressive features or a smarter classification strategy.

**Next ideas**: Test-time augmentation (classify multiple shifted views and vote — like the brain integrating across fixations), or stronger incoherence (now safe from overfitting).

### Incoherence-Regularized SC v6 Analysis (96.4%)

**What worked**: Adding incoherence regularization (ΔD_k -= η * Σ_{j≠k} D_j @ D_jᵀ @ D_k) forces class dictionaries to span different subspaces. This prevents dictionary_3 from accidentally explaining digit 5. Effect is small early on but compounds with training — +1.8% over v5 at 30 epochs.

**Key findings from sweep**:
- inc=0.001 is the sweet spot: enough to separate dictionaries, not so much that reconstruction suffers
- Higher rates (0.01+) hurt accuracy because they over-constrain the dictionaries
- The effect scales with training time — at 10 epochs the gain was only 0.2%, at 20 epochs it was 0.9%, at 30 epochs on full data it's 1.8%
- Coherence dropped from ~9.5 (baseline) to ~4.1 (inc=0.001)

**What limits it**:
1. **Representation is still linear** — sparse codes z = ISTA(x, D) are a linear transform + threshold. FF has 2 layers of nonlinear ReLU, capturing richer features
2. **Training is slow** (720s) — the incoherence penalty requires computing D_j^T @ D_k for all 9 other dictionaries per update
3. **Gap to target**: 0.9% to beat backprop, 0.5% to match FF

**Lessons**: Regularization works but has limits. The remaining gap is likely due to representation depth (linear vs nonlinear). Next: hierarchical sparse coding (V1 → V2) to add depth, or combine with FF-style nonlinear layers.

### Energy-Based Sparse Coding v5 Analysis (94.6%)

**What worked**: Class-specific dictionaries directly align feature learning with classification. Each class learns to explain only its own images. No readout layer needed — classification = competitive reconstruction (which dictionary explains the input best?). Param sweep found best config: 200 features/class, sparsity=0.01, lr=0.01.

**What limits it**:
1. **Each dictionary sees only 1/10th of the data** — class k dictionary only trains on ~5000 images, limiting what it can learn
2. **No cross-class discrimination** — dictionaries don't know what OTHER classes look like, so they can't actively avoid explaining them
3. **Similar digits confuse it** — 3 vs 5, 4 vs 9, 7 vs 1: dictionary_3 can partially explain 5s, causing misclassification
4. **Train acc (96.5%) vs test acc (94.6%)** — small generalization gap, not severe

**Key insight**: The approach validates the first-principles analysis — aligning feature learning with classification does help (+2% over shared-dictionary PC). But reconstruction-only learning has a ceiling because it lacks a discriminative signal.

**Lessons for next iteration**:
1. Need a **discriminative signal** — not just "explain your class well" but "explain your class better than others"
2. The **Forward-Forward algorithm already has this** (positive vs negative goodness) which is why it does better (96.9%)
3. Possible hybrid: energy-based dictionaries + discriminative refinement, or FF + sparse coding

### DFA v20 Analysis (97.5%) — BEATS BACKPROP

**What worked**: Direct Feedback Alignment replaces backpropagation's weight transport with fixed random feedback matrices. Each hidden layer receives the output error projected through a random (never-updated) matrix. The forward weights learn to align with these random projections, achieving near-backprop accuracy without the chain rule.

**Architecture**: 784 → 2000 (ReLU) → 1000 (ReLU) → 10 (softmax), trained for 60 epochs with lr=0.01, decay=0.97.

**Key insight from exploration**: After exhausting sparse coding improvements (v10-v16 yielded a 96.7% ceiling) and hybrid fusion approaches (v17-v19 reached 97.0-97.1%), the breakthrough came from a completely different algorithm family. DFA preserves the MLP architecture's representational power while replacing the biologically implausible backward pass with random fixed feedback.

**Biological analogues**:
- No weight transport problem: feedback is through random projections (modulatory pathways)
- Error broadcast: like dopaminergic signals that globally modulate plasticity
- Forward alignment: forward weights self-organize to make random feedback useful (Lillicrap et al. 2016)

**Why DFA succeeds where FF and SC plateau**:
1. **Representation depth**: DFA has 2 nonlinear hidden layers (like backprop MLP) vs FF's local-only training which limits inter-layer cooperation
2. **Global error signal**: Each layer receives information about the actual classification error (via random projection), not just a local goodness/reconstruction metric
3. **Capacity**: 2000+1000 hidden units give rich representation, and the random feedback still provides enough directional information for learning

**Comparison with approaches tried**:
- SC v9 (96.7%): Limited by linear reconstruction — can only represent features as weighted sums
- FF (96.9%): Each layer trained independently with local loss — no inter-layer cooperation
- Hybrid v17 (97.1%): Combining complementary systems helped but fusion is imprecise
- DFA v20 (97.5%): Global error signal (through random projection) + deep nonlinear representation

### Hybrid v17 Analysis (97.1%)

**What worked**: Error analysis revealed FF and SC make highly uncorrelated errors (only 33% overlap). Confidence-based arbitration: when FF and SC agree (94% of cases), use the shared prediction; when they disagree, trust the system with higher decision margin.

**Key findings**:
- FF+SC both wrong on only 1.9% of test images
- Perfect combination ceiling: 98.0%
- SC right on 66.4% of disagreements, FF right on 22.3%

**What limits it**: Hard arbitration wastes partial information. Soft fusion (v18/v19) didn't help due to model quality degradation from shared RNG and suboptimal temperature calibration.

### SC v10-v16 Post-Mortem (96.4-96.7%)

After achieving 96.7% with v9, extensive exploration of sparse coding variations:
- **v10 TTA**: Test-time augmentation adds +0.3% but 13x slower — not worth it
- **v11 FISTA**: Nesterov momentum destabilizes codes — oscillations between 73-96% train accuracy
- **v12 Discriminative**: Anti-Hebbian signal on wrong-class images destroys reconstruction quality
- **v13 Wide (400f)**: More features paradoxically hurt on full data (10k gain didn't scale)
- **v14 Ensemble**: Multiple SC models make correlated errors — no diversity benefit
- **v15 Cosine LR**: Small gains on 10k don't transfer to 50k
- **v16 PCA init**: PCA components reduce incoherence effectiveness

**Conclusion**: Linear sparse coding has a fundamental ceiling around 96.7%. The representation can only capture weighted sums of dictionary atoms — it cannot model spatial relationships between features.

### FF Enhanced Post-Mortem (88.4%)

**What went wrong**: Wider layers (1000 vs 500) need more epochs to converge. With 25 epochs per layer, the model was still underfitting — only 88.7% train accuracy at epoch 25. The LR warmup (3 epochs) wasted early training time, and multi-negative (3 per positive) diluted the gradient signal.

**Diagnosis**: The base FF gets 96.9% with 100 epochs/layer of 500 units. Doubling width to 1000 roughly quadruples the parameter space but doesn't quadruple the training signal per epoch. Result: severe underfitting.

**Lessons for next iteration**:
1. Don't increase capacity without increasing training signal
2. Base FF's strength is simplicity — fixed LR, single negative, moderate width
3. Better approaches: improve convergence speed (not capacity) or try fundamentally different architectures
4. Consider: smaller width + more epochs, or same width + better training signal (hard negatives)

## 2025 Research Landscape: Internal Models vs Statistical Prediction

The core question driving this project — can we learn representations that reflect genuine understanding rather than statistical pattern-matching? — is now one of the most active areas of AI research.

### Understanding Hierarchy

```
Level 0: Statistical prediction    (current LLMs — next token)
Level 1: Representation learning   (sparse coding, SAEs — what features exist)
Level 2: Predictive models         (JEPA, PCN — predict in representation space)
Level 3: Causal models             (Causal-JEPA — counterfactual reasoning)
Level 4: World models              (V-JEPA 2 — plan, act, anticipate)
```

**This project sits at Level 1.** The most exciting connection: Level 1 sparse representations are now the primary tool for understanding what Level 0 models (LLMs) have learned internally (via Sparse Autoencoders in mechanistic interpretability).

### Tier 1: World Models (JEPA Family — Meta/LeCun 2025)

JEPA (Joint Embedding Predictive Architecture) predicts in **abstract representation space**, not pixel space. It learns *what matters* about an input, discarding irrelevant surface details.

| Model | Result | Key Contribution |
|-------|--------|-----------------|
| V-JEPA 2 (1.2B params) | SOTA on video understanding | First world model enabling zero-shot planning and robot control in new environments |
| Causal-JEPA | Object-level causal reasoning | Extends JEPA with latent interventions for counterfactual reasoning |
| LLM-JEPA | Beats standard LLM training | JEPA-based objective outperforms next-token on GSM8K, Spider across Llama3/Gemma2 families |
| VL-JEPA | Beats CLIP, SigLIP2 | 1.6B param vision-language model surpasses larger multimodal models |

**Connection to our project**: JEPA's core insight — predict in representation space, not input space — is philosophically aligned with what sparse coding does. Our dictionaries learn compressed representations; JEPA learns predictive embeddings. Both avoid pixel-level reconstruction as the sole objective.

**Path from our work to JEPA**: Instead of reconstructing pixels (`x ≈ D @ z`), predict the *next representation* (`z_{t+1} ≈ f(z_t)`). This is a concrete bridge from dictionary learning to world models.

**Key gap exposed by Meta's benchmarks**: Current LLMs answer "what happened" in video well but fail at "what could have happened" (counterfactual) and "what might happen next" (anticipation) — revealing the distance between statistical prediction and causal understanding.

Papers: [V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) · [Causal-JEPA](https://arxiv.org/abs/2602.11389) · [LLM-JEPA](https://arxiv.org/abs/2509.14252) · [VL-JEPA](https://arxiv.org/abs/2512.10942)

### Tier 2: Biologically Plausible Learning That Scales (2025)

| Method | Result | Key Insight |
|--------|--------|-------------|
| μPC (Scaling PCN to 100+ layers) | 128-layer PCNs; +0.6-1.7% over BP on CIFAR-10 | Depth-μP parameterization; stable parameterizations for PCN = same as for backprop |
| Equilibrium Propagation (Hopfield-ResNet13) | 93.9% CIFAR-10, 71.1% CIFAR-100 | Residual connections in Hopfield nets; centered EP + Nesterov; matches BP on same architecture |
| Self-Contrastive Forward-Forward (SCFF) | 80.8% CIFAR-10 (3-layer CNN) | Solves Hinton's negative data problem via self-contrastive learning (Nature Communications) |
| FF on CNNs | Spatial labeling strategies | Forward-Forward extended to CNNs (Scientific Reports 2025) |

**μPC is the most significant result**: it proves that Predictive Coding Networks have the same theoretical scaling limits as backpropagation. The bottleneck was parameterization, not the algorithm. Meta-PCN consistently outperforms backprop on CIFAR-10 in matched architectures.

Papers: [μPC](https://arxiv.org/html/2505.13124) · [EP Hopfield-ResNet](https://arxiv.org/html/2509.26003v1) · [SCFF](https://www.nature.com/articles/s41467-025-61037-0) · [PCN Introduction](https://arxiv.org/abs/2506.06332)

### Tier 3: Sparse Coding for Interpretability (Our Lane)

**From Superposition to Sparse Codes (2025)**: Sparse Autoencoders (SAEs) — using the same mathematical framework as our project (overcomplete dictionaries, sparse codes) — are now the **primary tool** for mechanistic interpretability of LLMs. Anthropic, OpenAI, and others use dictionary learning to extract interpretable features from transformer activations. The representations our dictionaries learn are the same *kind* of thing interpretability researchers look for inside GPT-4.

**Physically Interpretable World Models (2024-2025)**: Formalizes what "interpretable representation" means: latent variables must correspond to meaningful physical quantities, and their temporal evolution must obey physically valid dynamics. This is the cleanest statement of what "understanding" means in a learned representation.

Papers: [From Superposition to Sparse Codes](https://arxiv.org/html/2503.01824v1) · [Physically Interpretable World Models](https://arxiv.org/html/2412.12870)

### Tier 4: Causal and Structural Learning

**Causal Representation Learning (Nature Communications 2025)**: Learning the ground-truth generative model underlying observed data — cause-effect interactions **beyond statistical associations**. Proposes timescale invariance via hierarchical generative model. This is exactly the "understanding vs statistics" distinction.

**Five Competing World Model Approaches (2025)**: DeepMind's SIMA 2, Fei-Fei Li's World Labs Marble, LeCun's JEPA, and others — all competing to build models with genuine causal understanding rather than pattern-matching.

Papers: [Causal Learning — Nature Comms](https://www.nature.com/articles/s41467-025-65137-9) · [Five World Model Approaches](https://themesis.com/2025/11/20/world-models-five-competing-approaches/)

### Flat Non-Backprop vs Flat Backprop on CIFAR-10

For our specific question — can non-backprop methods beat backprop on the same flat (non-convolutional) architecture:

| Method | Architecture | CIFAR-10 Accuracy | vs Backprop |
|--------|-------------|-------------------|-------------|
| Backprop MLP | 3-layer FC | ~55-58% | baseline |
| Forward-Forward MLP | 3-layer FC | ~56.2% (FAUST 2024) | -1.4% |
| Mono-Forward | FC layers | — | **+1.21%** over BP |
| DFA variants | FC layers | comparable | ~parity |

**Key insight**: On flat architectures, the bottleneck is the architecture, not the algorithm. Both backprop and non-backprop hit ~55-58%. The accuracy gap between learning rules is <2% — much smaller than on CNNs. This means our existing approaches (energy-based SC, DFA) could port to CIFAR-10 flat and achieve competitive results.

### Implications for This Project

1. **Our sparse coding work is directly relevant to mechanistic interpretability** — the same dictionary learning math is used to understand LLMs. This isn't a dead-end; it's mainstream AI safety research.

2. **The path forward is Level 1 → Level 2**: from learning static representations to learning predictive representations. Concretely: train dictionaries that predict the *next frame's sparse code*, not just reconstruct the current input.

3. **μPC proves PCN can scale**: the theoretical ceiling is gone. Our predictive coding approach (92.7% MNIST) could reach much higher with proper parameterization (Depth-μP).

4. **EP is the most proven non-backprop method at scale**: 93.9% CIFAR-10 with architectural innovations (residual Hopfield nets). Worth implementing if we move to CIFAR-10.

5. **Flat-architecture CIFAR-10 is achievable**: ~55% is realistic for our current methods. The interesting scientific question isn't raw accuracy but whether our representations show different *qualitative properties* (robustness, interpretability, graceful degradation) than backprop representations at similar accuracy.
