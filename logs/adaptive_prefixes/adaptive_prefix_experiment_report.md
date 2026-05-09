# Adaptive Prefix Stress Test: Analysis Report

## Setup

We compare the augmented Llama-3.2-3B model against two harmful-prefill regimes:

- **Static harmful prefix**: first `k` tokens of the original harmful answer from `Harmful-HEx-PHI.jsonl`.
- **Adaptive mined prefix**: first `k` tokens of prefixes mined against the augmented model in `mined_augmented_tinker.jsonl`.

The central question is whether augmentation against regular harmful prefilling leaves residual brittleness under adaptive, model-specific prefix search.

## ASR Results

The augmented model was evaluated on a matched set of 163 mined-prefix instructions. The static condition uses the original harmful HEx-PHI answer for each same instruction; the adaptive condition uses the mined prefix for that same instruction.

| Condition | k | Success / n | ASR | 95% Wilson CI |
|---|---:|---:|---:|---:|
| Base model, static harmful prefix | 10 | 308 / 330 | 93.3% | [90.1%, 95.6%] |
| Augmented model, static harmful prefix | 10 | 25 / 330 | 7.6% | [5.2%, 10.9%] |

The augmentation is clearly effective against the original static harmful-prefix attack: base static ASR at `k=10` is 93.3%, while augmented static ASR at `k=10` is 7.6%.

On the matched 163-instruction subset:

| k | Static success / n | Static ASR | Adaptive success / n | Adaptive ASR | Adaptive - static |
|---:|---:|---:|---:|---:|---:|
| 5 | 21 / 163 | 12.9% | 6 / 163 | 3.7% | -9.2 pp |
| 10 | 20 / 163 | 12.3% | 13 / 163 | 8.0% | -4.3 pp |
| 20 | 28 / 163 | 17.2% | 25 / 163 | 15.3% | -1.8 pp |
| 40 | 24 / 163 | 14.7% | 42 / 163 | 25.8% | +11.0 pp |

The matched result is more nuanced than a simple adaptive-dominates-static story. Static prefixes are stronger at short prefix lengths (`k=5,10,20`), while adaptive prefixes overtake static at `k=40`. Pairwise overlap by instruction shows the same pattern:

| k | Both succeed | Static-only | Adaptive-only | Neither | McNemar exact p |
|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 20 | 5 | 137 | 0.004 |
| 10 | 5 | 15 | 8 | 135 | 0.210 |
| 20 | 3 | 25 | 22 | 113 | 0.771 |
| 40 | 6 | 18 | 36 | 103 | 0.020 |

Thus, adaptive mining does not uniformly improve attack success. Its behavioral advantage emerges at the longest tested prefix budget, where adaptive produces 36 adaptive-only successes versus 18 static-only successes.

## Prefix Trajectory Analysis

The prefix-internal trajectory run measures next-token distributions and final-layer safety projections at different positions within the prefix. KL is measured against the static harmful-prefix distribution.

At the prefix endpoint (`lookback=0`), adaptive prefixes have large KL from static across all tested `k`:

| k | KL(adaptive || static) | Safety projection | Refusal margin |
|---:|---:|---:|---:|
| 5 | 11.45 | -1.33 | 1.06 |
| 10 | 11.59 | -1.64 | 1.20 |
| 15 | 12.53 | -0.97 | 1.35 |
| 20 | 13.83 | -1.63 | 1.16 |

Paired per-example comparisons show adaptive KL to static is positive for 163/163 examples at each endpoint. This strongly supports that adaptive prefixes induce a different immediate next-token distribution than regular harmful prefixes.

However, adaptive safety projection remains near static and far from refusal. Across endpoint measurements, adaptive projection is small or negative while refusal projection is large. This suggests adaptive prefixes are not simply moving representations along the measured static-to-refusal safety direction.

## Post-Prefix Recovery Analysis

The response-step trajectory run measures the model after generating additional tokens following the prefill. This tests whether the model recovers toward the static harmful-prefix regime, refusal regime, or remains in a distinct adaptive continuation regime.

Adaptive KL to static remains large and generally increases across generated response steps:

| k | KL at step 0 | KL at step 16 | Change |
|---:|---:|---:|---:|
| 5 | 11.45 | 14.97 | +3.52 |
| 10 | 11.59 | 13.76 | +2.17 |
| 15 | 12.52 | 13.90 | +1.38 |
| 20 | 13.83 | 14.49 | +0.66 |

This is the strongest trajectory signal:

> Adaptive prefixes induce a persistent output-distribution shift relative to regular harmful prefixes, and this shift does not disappear after generation begins.

Safety projection again remains small relative to refusal. At response step 16, adaptive projection as a fraction of refusal projection is:

| k | Adaptive / refusal projection |
|---:|---:|
| 5 | 6.4% |
| 10 | -8.5% |
| 15 | -12.9% |
| 20 | -5.0% |

Thus, the output-distribution shift is not explained by movement toward the refusal direction. The model is moving somewhere else in representation space, or shifting logits in a way not captured by this one-dimensional safety projection.

## Refusal Margin and Success Stratification

Refusal margin is defined as:

`log P(refusal tokens) - log P(compliance tokens)`.

The aggregate refusal-margin signal is mixed. Adaptive refusal margin from response step 0 to 16:

| k | Step 0 | Step 16 | Change |
|---:|---:|---:|---:|
| 5 | 1.06 | 2.37 | +1.31 |
| 10 | 1.20 | 1.58 | +0.37 |
| 15 | 1.35 | 1.28 | -0.07 |
| 20 | 1.16 | 0.97 | -0.20 |

For small `k`, the model becomes more refusal-biased after generation begins. For larger `k`, the refusal margin is flatter or slightly lower.

Success-conditioned analysis is more informative. For `k=20`, successful adaptive attacks have lower refusal margin than failed ones at every measured response step:

| Response step | Success mean | Failure mean | Success - failure |
|---:|---:|---:|---:|
| 0 | 0.75 | 1.24 | -0.49 |
| 1 | 0.84 | 1.39 | -0.55 |
| 2 | 0.86 | 1.37 | -0.52 |
| 4 | -0.01 | 1.00 | -1.02 |
| 8 | 0.54 | 1.66 | -1.13 |
| 16 | 0.43 | 1.06 | -0.64 |

This supports the interpretation that successful adaptive attacks are associated with weaker refusal pressure during continuation. The effect is clearest at `k=20`; at smaller `k`, success counts are low and success-conditioned signals are noisy.

## Main Claims Supported

1. **Safety augmentation substantially reduces regular harmful-prefix vulnerability.**  
   Static harmful-prefix ASR drops from 93.3% on the base model to 7.6% on the augmented model at `k=10`.

2. **Adaptive mined prefixes expose residual brittleness at longer prefix budgets.**  
   On the matched 163-instruction subset, adaptive ASR rises with prefix length and exceeds matched static ASR at `k=40` by 11.0 percentage points.

3. **Adaptive prefixes are distributionally distinct from static harmful prefixes.**  
   KL(adaptive || static) is large across examples and persists through generated continuation steps.

4. **The adaptive effect is not captured by a simple refusal-vs-static hidden-state direction.**  
   Safety projection remains close to static and far from refusal despite large output-distribution shifts.

5. **Successful adaptive attacks are associated with lower refusal margins, especially at larger k.**  
   At `k=20`, successful examples maintain lower refusal margin than failures across response steps.

## Interpretation

The strongest defensible interpretation is not that adaptive prefixes globally move hidden states into a new safety regime. Instead:

> Adaptive prefixes create a persistent output-distribution shift relative to regular harmful prefixes, while remaining close to the regular harmful-prefix regime along the measured refusal-vs-static hidden-state direction.

This suggests that adaptive prefix mining may exploit directions or token-competition effects not captured by a simple linear safety projection. In practical terms, augmentation against static harmful prefixes does not fully cover adaptive, model-specific prefix distributions.

## Limitations

- The strongest ASR is 25.8%, so this should be framed as residual brittleness, not a complete break.
- Adaptive prefixes do not dominate static prefixes at short prefix lengths. The strongest behavioral evidence is concentrated at `k=40`.
- Keyword ASR is a coarse evaluator and can be conservative or brittle.
- The safety projection is only one hidden-state direction. It does not rule out large hidden-state changes orthogonal to the static-refusal axis.
- Success-conditioned analysis has limited power at small `k` because there are few successful attacks.
