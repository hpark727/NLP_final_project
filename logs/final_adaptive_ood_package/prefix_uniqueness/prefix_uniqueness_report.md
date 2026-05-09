# Prefix Uniqueness Diagnostic

## Question

Are adaptive prefixes measurably different from matched static harmful prefixes, or are the recovery gains mainly because the model was fine-tuned on the adaptive examples?

## Setup

- Training diagnostic set: `163` matched static/adaptive prefix pairs from the original adaptive mining run.
- Heldout-style check: `23` static/adaptive prefix pairs from the gen3 hard set.
- Label: `adaptive=1`, `static=0`.
- Classifier: standardized logistic regression over simple surface features.

This is intentionally cheap and interpretable. It does not prove causality, but it tells us whether adaptive prefixes have measurable surface structure.

## Classifier Result

- 5-fold CV accuracy on matched 163-pair set with all features: `0.997`
- 5-fold CV AUC on matched 163-pair set with all features: `1.000`
- Gen3 hard-set accuracy with all features: `1.000`
- Gen3 hard-set AUC with all features: `1.000`
- Features used: `23`

![Score distributions](figures/prefix_classifier_score_distributions.png)

## Ablation Check

Because special-token artifacts are an easy separator, I reran the classifier after removing feature groups:

| Feature set | Features | CV accuracy | CV AUC | Gen3 accuracy | Gen3 AUC |
|---|---:|---:|---:|---:|---:|
| `all_features` | 23 | 0.997 | 1.000 | 1.000 | 1.000 |
| `no_special_tokens` | 19 | 0.972 | 0.993 | 0.913 | 0.989 |
| `no_special_or_length_format` | 13 | 0.963 | 0.989 | 0.891 | 0.938 |
| `semantic_overlap_only` | 3 | 0.761 | 0.798 | 0.543 | 0.586 |


The key point is not just that special tokens separate adaptive prefixes. Even after removing special-token features, the classifier remains highly separable. With only semantic/overlap features, separability drops, which suggests the distinction is mostly in generation/format/state features rather than simply semantic harmfulness.

## What Differs

Largest standardized mean differences, adaptive minus static:

| Feature | Static mean | Adaptive mean | Standardized delta |
|---|---:|---:|---:|
| `begin_token_count` | 0.000 | 2.000 | 2.00 |
| `starts_with_special` | 0.000 | 1.000 | 2.00 |
| `ends_mid_sentence` | 0.098 | 0.933 | 1.67 |
| `punct_char_fraction` | 0.050 | 0.104 | 1.19 |
| `char_len` | 1125.779 | 176.350 | -1.18 |
| `token_len` | 156.331 | 26.331 | -1.17 |
| `line_count` | 17.215 | 2.233 | -1.01 |
| `instruction_overlap` | 0.399 | 0.203 | -0.96 |


![Feature deltas](figures/prefix_feature_standardized_deltas.png)

Largest logistic weights after standardization:

| Feature | Coefficient | Adaptive minus static |
|---|---:|---:|
| `begin_token_count` | 1.86 | 2.000 |
| `starts_with_special` | 1.86 | 1.000 |
| `ends_mid_sentence` | 0.74 | 0.834 |
| `punct_char_fraction` | 0.54 | 0.054 |
| `line_count` | -0.43 | -14.982 |
| `repeated_bigram_fraction` | 0.39 | 0.032 |
| `token_len` | -0.34 | -130.000 |
| `char_len` | -0.33 | -949.429 |


![Classifier coefficients](figures/prefix_feature_classifier_coefficients.png)

## Interpretation

The classifier result supports a modest but important claim:

> Adaptive prefixes are measurably different from matched static harmful prefixes, even when prompt identity is controlled.

The most visible differences are surface/formatting and generation-state artifacts: special tokens, repetition, prefix length/line structure, list/code markers, and whether the prefix ends in an unfinished continuation. These are exactly the kinds of features one would expect if adaptive mining is exploiting model-conditioned continuation states rather than simply selecting semantically more harmful text.

## Caveat

This does not prove that these features cause jailbreak success. It only shows that adaptive and static prefixes are separable. The causal version would require feature-matched controls or prefix-swap experiments:

- Train on adaptive prefixes from one prompt split and test on newly mined adaptive prefixes from held-out prompts.
- Match static and adaptive prefixes on length, category, repetition, and special-token count.
- Swap adaptive prefixes onto different harmful prompts and test whether the attack transfers.
