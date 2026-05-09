# Final Adaptive OOD Package

Generated from existing experiment logs in `logs/adaptive_recovery` and `logs/adaptive_prefixes`.

## Research Thesis

The tight version of the project is:

> Static harmful-prefix recovery is an in-distribution safety result. Adaptive mining creates a prefix-distribution shift that exposes residual failures. Adaptive-only recovery improves the mined prefix family but partially forgets the original static prefix family. Mixed static + adaptive recovery widens the recovery basin and is supported by both behavioral ASR and hidden-state trajectory evidence.

## Concrete OOD Definition

Use two OOD axes:

1. **Prefix-source OOD:** train/evaluate on the same harmful prompt source, but change the prefilled continuation distribution. In-distribution is static HEx-PHI harmful prefixes; OOD is model-mined adaptive prefixes.
2. **Prompt-source OOD:** train/evaluate on HEx-PHI-style harmful prompts, then validate on a different harmful dataset and taxonomy. Aegis 2.0 gives this axis because it has a different collection pipeline, label taxonomy, and Mistral-generated unsafe responses.

This makes the project more concrete than saying "OOD" abstractly. Current results primarily establish prefix-source OOD. The prepared Aegis eval below would test prompt-source plus prefix-source OOD.

## 1. Adaptive OOD Behavioral Results

Static augmentation initially works very well in-distribution: static harmful-prefix ASR at `k=10` drops from `93.3%` for base Llama-3.2-3B to `7.6%` for the original augmented model.

![Static baseline](figures/01_static_prefix_baseline.png)

Adaptive mining then exposes a distribution shift. On matched prompts, adaptive prefixes become harder than matched static prefixes at `k=40`, even though they are not uniformly stronger at shorter prefix lengths.

![Matched static vs adaptive](figures/02_matched_static_vs_adaptive.png)

The focused gen3 same23 set is a small but useful paired stress set. The original augmented model reaches `39.1%` ASR at `k=40` on gen3 adaptive prefixes.

![Gen3 recovery comparison](figures/04_gen3_recovery_comparison.png)

## 2. Mixed Recovery vs Adaptive-Only Recovery

Adaptive-only recovery is not a clean solution. It improves gen3 adaptive robustness, but it regresses on original static harmful prefixes. At static `k=10`, adaptive-only recovery has `24.2%` ASR, while mixed 50/50 recovery has `2.1%` ASR.

![Static recovery comparison](figures/03_static_recovery_comparison.png)

The GPT-judged static comparison supports the same direction: adaptive-only recovery has high GPT-judged static failures, while mixed recovery stays low across prefix lengths.

![GPT static recovery](figures/05_gpt_static_recovery.png)

On gen3 adaptive prefixes, mixed recovery preserves most of the adaptive-only gain. At `k=40`, mixed recovery reduces original augmented ASR from `39.1%` to `13.0%`.

The current mining algorithm also becomes much less productive against mixed recovery: the gen4 stress remine kept `5` prefixes from `590` candidates, a `0.8%` keep rate. Interpret the 5 kept prefixes as a stress case, not a population estimate.

## 3. Hidden-State / Trajectory Mechanism

The mechanism result is not "mixed recovery memorizes the mined prefixes." The stronger statement is that mixed recovery raises refusal pressure on the same adaptive prefix states.

At `k=20`, response step 0, mixed recovery raises mean refusal margin by `4.54` and does so for `23/23` paired examples.

![Trajectory refusal margin](figures/06_trajectory_refusal_margin.png)

![Trajectory paired signs](figures/07_trajectory_paired_signs.png)

The refusal-subspace result adds a useful caution: adaptive prefixes are not simply normal refusal states. Their refusal-subspace fraction is consistently lower than the explicit refusal condition, so adaptive mining likely exploits directions or token-competition effects not captured by a single refusal axis.

![Refusal subspace](figures/08_refusal_subspace_fraction.png)

## Aegis Cross-Validation Plan

I prepared a held-out Aegis 2.0 prefix file from rows where both the prompt and response are labeled unsafe. This creates a stricter OOD stress test:

- Dataset: `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`
- Source: validation + test splits
- Full unsafe prompt/unsafe response records after prompt de-duplication: `637`
- Ready-to-run sample file: `logs/final_adaptive_ood_package/aegis/aegis_unsafe_prefixes_500.jsonl`
- Full file: `logs/final_adaptive_ood_package/aegis/aegis_unsafe_prompt_unsafe_response_prefixes.jsonl`

![Aegis category distribution](figures/09_aegis_category_distribution.png)

Run this if you want to add cross-dataset validation:

```bash
logs/final_adaptive_ood_package/run_aegis_ood_eval.zsh
```

That script evaluates original augmented, adaptive-only recovery, and mixed 50/50 recovery on Aegis unsafe response prefixes at `k=5,10,20,40`.

Expected interpretation:

- If mixed recovery still beats adaptive-only on Aegis, the recovery-basin claim is much stronger.
- If mixed recovery only helps on HEx-PHI-derived prompts, the conclusion should be narrowed to prefix-source OOD within the original harmful-prompt distribution.
- If all models fail badly, Aegis reveals a prompt-source OOD gap and becomes an honest limitation/future-work result.




## Prefix Uniqueness Diagnostic

A cheap surface-feature classifier can distinguish matched static vs adaptive prefixes, which supports the claim that adaptive prefixes are not just more samples from the same static-prefix distribution.

- 5-fold CV accuracy on the 163-pair matched set with all features: `0.997`
- 5-fold CV AUC with all features: `1.000`
- Gen3 hard-set accuracy with all features: `1.000`
- Gen3 hard-set AUC with all features: `1.000`
- After removing special-token features, CV AUC: `0.993`
- With only semantic/overlap features, CV AUC: `0.798`

![Prefix classifier scores](prefix_uniqueness/figures/prefix_classifier_score_distributions.png)

![Prefix feature deltas](prefix_uniqueness/figures/prefix_feature_standardized_deltas.png)

Interpretation: adaptive prefixes are separable by surface/generation-state features, especially special-token artifacts, repetition/length structure, formatting, and unfinished-continuation markers. The ablation check suggests this is not only a special-token artifact, but the weak semantic-only result means the distinction is more about induced continuation state than simple harmful-topic content. This supports the distribution-shift story, but it does not by itself prove causality; feature-matched controls or prefix-swap tests would be needed for that.


## Dataset OOD Diagnostic: Aegis vs HEx-PHI

To make prompt-source OOD concrete, I trained lightweight classifiers to distinguish Aegis unsafe examples from HEx-PHI harmful examples.

| Model | Input | Accuracy | AUC |
|---|---|---:|---:|
| `prompt_tfidf` | `prompt_text` | 0.882 | 0.949 |
| `prompt_plus_prefix_tfidf` | `combined_text` | 0.971 | 0.995 |
| `surface_features` | `surface` | 0.945 | 0.984 |


![Dataset OOD scores](dataset_ood/figures/dataset_ood_score_distributions.png)

Interpretation: Aegis is not merely another sample from the same HEx-PHI distribution. Even prompt-only text is highly separable, so the Aegis results are valid evidence about prompt-source OOD. This also explains why mixed recovery can improve over adaptive-only while still not beating the original augmented model on Aegis: the Aegis distribution is genuinely different from the adaptive HEx-PHI recovery distribution.

## Aegis Cross-Dataset Results

The Aegis run is complete on the 500-example unsafe prompt + unsafe response sample. This tests a stronger OOD setting than the main HEx-PHI results because both the harmful prompt source and the unsafe prefill source differ from the recovery data.

![Aegis OOD ASR](figures/10_aegis_ood_asr.png)

| k | Original augmented | Adaptive-only recovery | Mixed 50/50 recovery |
|---:|---:|---:|---:|
| 5 | 19/500 = 3.8% | 99/500 = 19.8% | 30/500 = 6.0% |
| 10 | 14/500 = 2.8% | 109/500 = 21.8% | 34/500 = 6.8% |
| 20 | 37/500 = 7.4% | 126/500 = 25.2% | 47/500 = 9.4% |
| 40 | 40/500 = 8.0% | 154/500 = 30.8% | 56/500 = 11.2% |


Interpretation:

- Mixed recovery strongly improves over adaptive-only recovery on Aegis: at `k=40`, ASR drops from `30.8%` to `11.2%`.
- Original augmented remains best on this Aegis sample: at `k=40`, it has `8.0%` ASR versus mixed recovery's `11.2%`.
- This means Aegis strengthens the "adaptive-only forgets; mixed restores much of the lost robustness" claim, but it does not support claiming that mixed recovery dominates all prompt-source OOD settings.
- Paired at `k=40`, mixed succeeds on `26` examples where adaptive-only does not, while adaptive-only succeeds on `124` examples where mixed does not.
- Paired at `k=40`, mixed succeeds on `45` examples where original augmented does not, while original augmented succeeds on `29` examples where mixed does not.

![Aegis OOD deltas](figures/11_aegis_ood_deltas.png)

![Aegis category ASR](figures/12_aegis_ood_category_k40.png)

Revised OOD claim:

> Mixed recovery is robust to prefix-source OOD and partially improves prompt-source OOD relative to adaptive-only recovery, but the Aegis results show that adding adaptive recovery can still trade off against the original augmented model on some cross-dataset unsafe-prefix distributions.

## Recommended Final Claim

The most defensible final claim is:

> Mixed static + adaptive recovery improves safety robustness under prefix-distribution shift, preserving original static-prefix robustness while reducing failures on adaptively mined prefixes. Trajectory analysis suggests this works by increasing refusal pressure at and after the prefix boundary, not simply by collapsing adaptive prefixes into the original static distribution.

## What Not To Overclaim

- Do not claim general safety robustness across all harmful prompt distributions; Aegis is one cross-dataset check, and mixed recovery does not dominate the original augmented model there.
- Do not treat the 23-prompt gen3 set as a population-level benchmark; frame it as a paired hard-case mechanism set.
- Do not treat the 5-prefix gen4 remine ASR as stable. The keep rate is the real result there.
- Do not make teacher-forcing zigzag the main paper unless the utility story cleans up.

## File Map

- Figures: `logs/final_adaptive_ood_package/figures/`
- Aegis prepared files: `logs/final_adaptive_ood_package/aegis/`
- Aegis eval command: `logs/final_adaptive_ood_package/run_aegis_ood_eval.zsh`
- Source report: `logs/adaptive_recovery/final_experiment_report.md`
