# Causal Dictionary Learning — Lab Notebook

## Experiment Log

### Exp 0: Baseline (original encoding, ISTA, 30 atoms)
- **Result**: FAIL 0/5. Ratios 4.5-5.6, Jaccards 0.46-0.87, specialization 0.61.
- **Diagnosis**: 62-dim binary one-hot encoding has 50 dims for position (one-hot)
  but only 4 dims for action. Dictionary learns position reconstruction, not rule structure.
  Displacement (the causal signal) is not encoded — must be inferred from position pairs.

### Exp 1: Encoding sweep (ISTA, 30 atoms)
| Encoding | Ratios          | Spec  | Verdict |
|----------|-----------------|-------|---------|
| original | 6.0, 4.4, 0.5, 5.2, 2.4 | 0.62 | FAIL 0/5 |
| enriched | 5.9, 4.6, 0.5, 5.6, 2.4 | 0.62 | FAIL 0/5 |
| compact  | 1.1, 1.3, 0.8, 1.2, 1.2 | 0.69 | 1/5 (Jaccard fails) |

- **Finding**: Compact encoding (19 dims with displacement) **solves the ratio problem**.
  All ratios near 1.0 vs 4-6 with original encoding.
- **Why**: Displacement features directly encode the causal effect. Dictionary can factor
  displacement patterns instead of memorizing 50-dim position one-hots.

### Exp 2: Capacity sweep (compact encoding, ISTA)
| Atoms | Spec  | Ratios                    | Jaccard (T1) |
|-------|-------|---------------------------|-------------|
| 8     | 0.66  | 0.89, 1.75, 0.55, 1.18, 1.14 | 0.50 |
| 10    | 0.71  | 0.71, 1.56, 0.67, 1.43, 0.80 | 0.50 |
| 12    | 0.67  | 1.34, 1.97, 0.78, 2.28, 1.28 | 0.64 |
| 15    | 0.73  | 0.96, 1.49, 0.58, 1.80, 1.28 | 0.31 |
| 30    | 0.69  | 1.1, 1.3, 0.8, 1.2, 1.2      | 0.50 |
| 64    | 0.78  | 1.6, 1.7, 0.7, 1.7, 1.4      | 0.31 |

- **Finding**: 8-10 atoms is optimal. Fewer atoms forces atom reuse → higher Jaccard.
  More atoms → atoms specialize more (higher spec score) but Jaccard drops because
  gravity activates too many atoms, inflating the union.

### Exp 3: Architecture comparison (compact encoding, 64 atoms)
| Architecture | Spec  | Best ratio | Jaccard range |
|-------------|-------|-----------|---------------|
| ISTA        | 0.78  | 1.6       | 0.31-0.50     |
| NMF         | 0.74  | 7.1 (!)   | 0.11-0.22     |
| K-SVD       | 0.85  | 3.3       | 0.14-0.32     |
| Autoencoder | 0.73  | 2.5       | 0.59-0.69     |

- **Finding**: ISTA wins on ratios. K-SVD has best specialization but worst ratios
  (atoms overspecialize → can't compose). NMF is poor overall.
  Autoencoder has best Jaccard — nonlinear encoding distributes activations more evenly.

### Exp 4: ISTA 8 atoms across seeds (compact encoding)
| Seed | T1 J | T2 J | T3 J | T4 J | Pass |
|------|------|------|------|------|------|
| 42   | 0.38 | 0.75 | 0.71 | 0.88 | 4/5  |
| 123  | 0.50 | 0.88 | 0.71 | 0.88 | 4/5  |
| 7    | 0.62 | 0.75 | 0.50 | 0.75 | 3/5  |
| 999  | 0.38 | 0.75 | 0.57 | 0.88 | 3/5  |

- **Finding**: T1 (gravity+containment) Jaccard NEVER reaches 0.7 with ISTA.
  T2, T3, T4 regularly pass. The issue is structural: gravity activates 6-7 of 8 atoms.

### Exp 5: Autoencoder 6 atoms (compact encoding)
| Config | T1 R | T1 J | T2-T5 | Pass |
|--------|------|------|-------|------|
| ae6, lr=0.01, sp=0.05, s42 | 2.11 | 1.00 | all pass | 4/5 |
| ae6, lr=0.003, sp=0.1, s42 | 2.07 | 1.00 | all pass | 4/5 |
| ae6, lr=0.001, sp=0.05, s42, e500 | 2.05 | 1.00 | all pass | 4/5 |

- **Finding**: Autoencoder with 6 atoms gets PERFECT Jaccards (1.0 everywhere)
  because all 6 atoms are shared across rules (too few to specialize).
  But T1 ratio is 2.05-2.11 — barely over the 2.0 threshold.

### Exp 6: Decorrelated ISTA (lateral inhibition)
- **Finding**: No improvement over standard ISTA. Lateral inhibition doesn't help
  because the issue is the Jaccard metric, not atom co-activation.

### Key Insight
**4/5 is the reliable ceiling** with current setup. T1 gravity+containment is structurally
hard because:
1. Gravity is the most diverse rule (objects fall from many heights/positions)
2. This activates more atoms than containment or contact
3. The Jaccard |C ∩ (A∪B)| / |C ∪ (A∪B)| is dragged down by large |A|

Composition events use a SUBSET of single-rule atoms (100% precision at low threshold),
but Jaccard penalizes the excess atoms in the union.

### Exp 7: Grid search (compact encoding, ISTA + autoencoder)
- **Setup**: 160 configs, 5 seeds, two-phase (fast 500 events → full 2000 events on top 10)
- **Finding**: Very low sparsity (0.02) is optimal. Best config: ISTA 6 atoms sp=0.02, mean 3.8/5.
- **Conclusion**: Hyperparameter tuning within existing architectures hits diminishing returns.

### Exp 8: Novel architectures (compact encoding, 5 seeds each)
| Architecture | Config | Mean Pass | Best | Worst | Key observation |
|-------------|--------|-----------|------|-------|-----------------|
| ProductOfExperts | 3rule+3pos, sp=0.02 | 4.4/5 | 5/5 | 2/5 | **5/5 on seeds 42, 123, 999, 2024** |
| ContrastiveDictionary | 8a, cw=0.5, sp=0.02 | 4.4/5 | 5/5 | 4/5 | **Most consistent. 5/5 on seeds 42, 7** |
| ProductOfExperts | 4rule+4pos, sp=0.02 | 4.2/5 | 5/5 | 3/5 | 5/5 on seeds 42, 123, 7 |
| SlotDictionary | 6 slots | 4.0/5 | 4/5 | 4/5 | Perfect Jaccards, T1 ratio always ~2.02 |
| ISTA baseline | 8a, sp=0.02 | 3.8/5 | 5/5 | 3/5 | Never consistently 5/5 |

**Key findings**:
1. **ProductOfExperts** breaks the T1 barrier by factoring rule atoms from position atoms.
   Gravity gets a compact rule code instead of activating most atoms.
2. **ContrastiveDictionary** is the most reliable — worst case is still 4/5.
   The contrastive loss pushes atoms toward single-rule specialization.
3. **SlotDictionary** is perfectly consistent but T1 ratio plateaus at ~2.02 (just over threshold).
4. Both novel architectures that achieved 5/5 did so by addressing the core problem:
   gravity's positional diversity diluting rule-specific atom patterns.

### Key Insight (updated)
**5/5 is now achievable** with architectural innovation. The breakthrough was separating
rule representation from position representation (ProductOfExperts) or adding explicit
specialization pressure (ContrastiveDictionary). Standard ISTA hits a 4/5 ceiling because
atoms must jointly encode both rule type and spatial location.

## Best Configurations

1. **ContrastiveDictionary + compact + 8 atoms + sp=0.02 + cw=0.5**: Most reliable (worst=4/5, mean=4.4)
2. **ProductOfExperts + compact + 3rule+3pos + sp=0.02**: Highest peak (5/5 on 4/5 seeds, mean=4.4)
3. **ISTA + compact + 8 atoms + sp=0.02**: Solid baseline (mean=3.8/5)
