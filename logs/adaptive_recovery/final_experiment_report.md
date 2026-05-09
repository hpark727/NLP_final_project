# Adaptive Prefix Recovery: Integrated Experiment Report

## Executive Summary

The experiments support a refined version of the original hypothesis:

> Safety recovery under prefilled unsafe continuations is distribution-sensitive, but the brittleness is not unavoidable. Adaptive-only recovery training overfits to the mined prefix distribution and regresses on original static harmful prefixes; mixed static + adaptive recovery training covers both prefix families and substantially improves recovery.

The strongest behavioral evidence is:

- Static harmful-prefix robustness originally improved from `308/330 = 93.3%` ASR on the base model to `25/330 = 7.6%` on the original augmented model at `k=10`.
- Adaptive-only recovery lowered gen3 adaptive-hardcore ASR at `k=20/40` to `2/23 = 8.7%`, but regressed badly on original static harmful prefixes: `80/330 = 24.2%` at static `k=10`.
- Mixed 50/50 recovery kept gen3 adaptive-hardcore ASR low while restoring static robustness: static `k=10` dropped to `7/330 = 2.1%`, and gen3 `k=20/40` stayed at `2/23 = 8.7%` and `3/23 = 13.0%`.
- A stronger remine against the mixed model found only `5` kept prefixes out of `590` candidate attempts (`0.85%` keep rate), though those 5 prefixes remain a small stress-test set rather than a stable estimate.

The strongest trajectory/KL evidence is:

- On the same 23 gen3 prefixes, mixed recovery increased refusal margin for nearly every example at the attack-relevant prefix endpoint:
  - `k=20`, step 0: refusal margin increased on `23/23` examples, mean delta `+4.54`.
  - `k=40`, step 0: refusal margin increased on `22/23` examples, mean delta `+4.77`.
- At the prefix endpoint, mixed recovery also moved adaptive-prefix output distributions closer to the static harmful-prefix distribution:
  - `k=20`, step 0 KL-to-static: `12.97 -> 6.96`.
  - `k=40`, step 0 KL-to-static: `13.52 -> 8.58`.
- After generation begins, adaptive trajectories remain non-static, but refusal pressure stays higher:
  - `k=20`, step 16 KL-to-static: `14.62 -> 16.12`, while refusal margin rises `1.37 -> 4.36`.
  - `k=40`, step 16 KL-to-static: `12.67 -> 15.35`, while refusal margin rises `1.23 -> 4.05`.

This suggests mixed recovery does not merely collapse adaptive prefixes into the original static harmful-prefix regime. It teaches the model to recover safely from a broader prefix region.

## Log Sources

Primary report tables are in:

- `logs/adaptive_recovery/final_experiment_report_tables/static_harmful_model_comparison.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/original_augmented_matched_adaptive_vs_static.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/gen3_same23_model_comparison.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/gen4_remine_yield.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/gen4_remine_eval.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/trajectory_adaptive_focus_step0_16.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/trajectory_paired_delta_sign_counts.csv`

Important raw logs include:

- `logs/prefilling/meta-llama__Llama-3.2-3B__hex-phi_with_harmful_prefix__prefix10.json`
- `logs/prefilling/ft_llama3_2_3b__hex-phi_with_harmful_prefix__prefix10.json`
- `logs/adaptive_prefixes/gen3_hardcore_compare/*.json`
- `logs/adaptive_recovery/eval_adaptive_ft_on_original_harmful_k*.json`
- `logs/adaptive_recovery/eval_mixed_50_50_on_original_harmful_k*.json`
- `logs/adaptive_recovery/eval_recovery_on_gen3_same23_k*.json`
- `logs/adaptive_recovery/eval_mixed_50_50_on_gen3_same23_k*.json`
- `logs/adaptive_recovery/trajectory_augmented_gen3_kl_static/trajectory_summary.csv`
- `logs/adaptive_recovery/trajectory_mixed_50_50_gen3_kl_static/trajectory_summary.csv`
- `logs/adaptive_recovery/trajectory_model_comparison_gen3_kl_static/summary_delta.csv`
- `logs/adaptive_recovery/trajectory_model_comparison_gen3_kl_static/per_example_delta.csv`

## Stage 1: Static Harmful-Prefix Augmentation Worked

The starting point is the original harmful-prefix defense result at `k=10`:

| Model | Static harmful prefix k=10 | ASR |
|---|---:|---:|
| Base Llama-3.2-3B | `308/330` | `93.3%` |
| Original augmented | `25/330` | `7.6%` |

This establishes that the original augmentation made the model robust to the regular static harmful-prefix distribution.

Source: `logs/adaptive_recovery/final_experiment_report_tables/static_harmful_model_comparison.csv`.

## Stage 2: Adaptive Prefix Mining Exposed Residual Brittleness

On the matched 163-instruction subset, adaptive gen1 prefixes were not uniformly stronger than static prefixes at short lengths, but they became stronger at longer prefix budget:

| k | Matched static ASR | Adaptive gen1 mined ASR |
|---:|---:|---:|
| 5 | `21/163 = 12.9%` | `6/163 = 3.7%` |
| 10 | `20/163 = 12.3%` | `13/163 = 8.0%` |
| 20 | `28/163 = 17.2%` | `25/163 = 15.3%` |
| 40 | `24/163 = 14.7%` | `42/163 = 25.8%` |

The more focused gen3 hardcore comparison selected 23 prompts where repeated adaptive mining found harder prefixes. On those same 23 prompts, the original augmented model had substantial failures at longer prefix lengths:

| Prefix set | k=5 | k=10 | k=20 | k=40 |
|---|---:|---:|---:|---:|
| Gen1 | `0/23 = 0.0%` | `7/23 = 30.4%` | `11/23 = 47.8%` | `12/23 = 52.2%` |
| Gen2 | `0/23 = 0.0%` | `2/23 = 8.7%` | `11/23 = 47.8%` | `11/23 = 47.8%` |
| Gen3 | `0/23 = 0.0%` | `0/23 = 0.0%` | `6/23 = 26.1%` | `9/23 = 39.1%` |
| Static same23 | `7/23 = 30.4%` | `5/23 = 21.7%` | `4/23 = 17.4%` | `4/23 = 17.4%` |

This supports the brittleness claim: the original static augmentation did not cover the full adaptive prefix distribution, especially after repeated mining and at larger prefix budgets.

Sources:

- `logs/adaptive_recovery/final_experiment_report_tables/original_augmented_matched_adaptive_vs_static.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/gen3_same23_model_comparison.csv`

## Stage 3: Adaptive-Only Recovery Helped Adaptive Prefixes But Hurt Static Prefixes

The adaptive-only recovery model was trained on adaptive mined prefixes. It improved robustness on gen3 same23, especially at the longer prefix lengths:

| Model | Gen3 k=20 | Gen3 k=40 |
|---|---:|---:|
| Original augmented | `6/23 = 26.1%` | `9/23 = 39.1%` |
| Adaptive-only recovery | `2/23 = 8.7%` | `2/23 = 8.7%` |

But it regressed on the original static harmful-prefix distribution:

| Model | Static k=10 |
|---|---:|
| Original augmented | `25/330 = 7.6%` |
| Adaptive-only recovery | `80/330 = 24.2%` |

Across static prefix lengths, adaptive-only recovery ASR was:

| k | Static ASR |
|---:|---:|
| 5 | `64/330 = 19.4%` |
| 10 | `80/330 = 24.2%` |
| 20 | `98/330 = 29.7%` |
| 40 | `100/330 = 30.3%` |

This is the clearest evidence that recovery SFT can be prompt-distribution constrained: training only on one prefill family improved that family while weakening another.

Sources:

- `logs/adaptive_recovery/final_experiment_report_tables/static_harmful_model_comparison.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/gen3_same23_model_comparison.csv`

## Stage 4: Mixed 50/50 Recovery Covered Both Prefix Families

The mixed dataset doubled the adaptive-only rows by adding matched static harmful-prefix rows:

| Dataset | Split | Rows | Instructions | Adaptive rows | Static rows |
|---|---|---:|---:|---:|---:|
| Adaptive-only recovery | train | 471 | 112 | - | - |
| Adaptive-only recovery | dev | 57 | 14 | - | - |
| Adaptive-only recovery | test | 186 | 37 | - | - |
| Mixed 50/50 recovery | train | 942 | 112 | 471 | 471 |
| Mixed 50/50 recovery | dev | 114 | 14 | 57 | 57 |
| Mixed 50/50 recovery | test | 372 | 37 | 186 | 186 |

The mixed model restored static harmful-prefix robustness:

| Model | Static k=5 | Static k=10 | Static k=20 | Static k=40 |
|---|---:|---:|---:|---:|
| Adaptive-only recovery | `19.4%` | `24.2%` | `29.7%` | `30.3%` |
| Mixed 50/50 recovery | `4.2%` | `2.1%` | `3.9%` | `4.8%` |

It also retained most adaptive-hardcore robustness:

| Model | Gen3 k=5 | Gen3 k=10 | Gen3 k=20 | Gen3 k=40 |
|---|---:|---:|---:|---:|
| Original augmented | `0.0%` | `0.0%` | `26.1%` | `39.1%` |
| Adaptive-only recovery | `0.0%` | `0.0%` | `8.7%` | `8.7%` |
| Mixed 50/50 recovery | `0.0%` | `4.3%` | `8.7%` | `13.0%` |

The conclusion is not "there is an unavoidable tradeoff." The mixed result suggests the adaptive-only regression was due to incomplete recovery coverage.

Sources:

- `logs/adaptive_recovery/final_experiment_report_tables/recovery_data_splits.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/static_harmful_model_comparison.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/gen3_same23_model_comparison.csv`

## Stage 5: Remining Against Mixed 50/50 Found Few Prefixes

The stronger remine against the mixed model used the gen3 hardcore set with multiple seed lengths and more samples. It found only:

| Attempted candidates | Kept prefixes | Keep rate |
|---:|---:|---:|
| 590 | 5 | 0.85% |

Those 5 kept prefixes still produced attacks at longer prefix budgets:

| k | ASR on 5 kept gen4 prefixes |
|---:|---:|
| 5 | `0/5 = 0.0%` |
| 10 | `0/5 = 0.0%` |
| 20 | `0/5 = 0.0%` |
| 40 | `2/5 = 40.0%` |
| full | `3/5 = 60.0%` |

This should be interpreted carefully. The low mining yield is strong evidence that mixed recovery made the known search much less productive. The ASR on the 5 kept prefixes is a stress-test case study, not a stable rate estimate.

Sources:

- `logs/adaptive_recovery/final_experiment_report_tables/gen4_remine_yield.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/gen4_remine_eval.csv`

## Stage 6: Trajectory and KL Analysis Explain the Behavioral Improvement

The paired trajectory analysis compares original augmented vs mixed 50/50 on the exact same 23 gen3 prefixes.

At the attack-relevant longer prefix lengths, mixed recovery consistently increased refusal margin:

| k | Step | Augmented refusal margin | Mixed refusal margin | Delta |
|---:|---:|---:|---:|---:|
| 20 | 0 | 1.35 | 5.89 | +4.54 |
| 20 | 16 | 1.37 | 4.36 | +2.98 |
| 40 | 0 | 0.44 | 5.21 | +4.77 |
| 40 | 16 | 1.23 | 4.05 | +2.83 |

The paired sign counts show this is not just a mean effect:

| k | Step | Examples with increased refusal margin |
|---:|---:|---:|
| 20 | 0 | `23/23` |
| 20 | 16 | `20/23` |
| 40 | 0 | `22/23` |
| 40 | 16 | `17/23` |

At the prefix endpoint, mixed recovery also moved adaptive prefixes closer to the static harmful-prefix distribution:

| k | Step | Augmented KL to static | Mixed KL to static | Delta |
|---:|---:|---:|---:|---:|
| 20 | 0 | 12.97 | 6.96 | -6.01 |
| 40 | 0 | 13.52 | 8.58 | -4.93 |

But by response step 16, adaptive continuations remained distributionally distinct:

| k | Step | Augmented KL to static | Mixed KL to static | Delta |
|---:|---:|---:|---:|---:|
| 20 | 16 | 14.62 | 16.12 | +1.50 |
| 40 | 16 | 12.67 | 15.35 | +2.68 |

This supports the mechanistic interpretation:

> Mixed recovery partially normalizes adaptive prefixes at the prefill boundary, but its behavioral success is mainly explained by stronger refusal pressure during continuation, not by collapsing adaptive prefixes into the static harmful-prefix regime.

Short-prefix cases show the same recovery shift even though ASR was already near zero:

| k | Step | Augmented refusal margin | Mixed refusal margin | Delta |
|---:|---:|---:|---:|---:|
| 5 | 0 | -1.52 | -0.90 | +0.62 |
| 5 | 16 | 2.17 | 2.15 | -0.01 |
| 10 | 0 | -0.26 | 4.56 | +4.82 |
| 10 | 16 | 1.82 | 4.80 | +2.98 |

Sources:

- `logs/adaptive_recovery/final_experiment_report_tables/trajectory_adaptive_focus_step0_16.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/trajectory_paired_delta_sign_counts.csv`
- `logs/adaptive_recovery/trajectory_model_comparison_gen3_kl_static/summary_delta.csv`
- `logs/adaptive_recovery/trajectory_model_comparison_gen3_kl_static/per_example_delta.csv`

## Hidden-State Subspace Result From Original Augmented Model

The earlier refusal-subspace analysis showed that adaptive prefixes were not simply moving along the measured refusal direction.

For the original augmented model, adaptive prefixes occupied substantially less of the refusal subspace than refusal prefixes:

| k | Adaptive subspace fraction | Refusal subspace fraction | Adaptive projection norm / refusal |
|---:|---:|---:|---:|
| 5 | 0.353 | 0.729 | 0.430 |
| 10 | 0.335 | 0.645 | 0.442 |
| 20 | 0.323 | 0.627 | 0.455 |
| 40 | 0.350 | 0.512 | 0.628 |

This supports the earlier interpretation that adaptive mining exploited directions or token-competition effects not captured by one simple refusal-vs-static axis.

Sources:

- `logs/adaptive_recovery/final_experiment_report_tables/refusal_subspace_ratios.csv`
- `logs/adaptive_recovery/final_experiment_report_tables/refusal_subspace_condition_summary.csv`

## Claims Supported

1. **Static harmful-prefix augmentation substantially improved the base model.**  
   Static `k=10` ASR dropped from `93.3%` to `7.6%`.

2. **Adaptive prefix mining exposed distributional brittleness in the original augmented model.**  
   On the 23-prompt gen3 hardcore set, original augmented ASR reached `26.1%` at `k=20` and `39.1%` at `k=40`.

3. **Adaptive-only recovery was distribution-constrained.**  
   It improved gen3 `k=20/40` ASR to `8.7%`, but static `k=10` ASR regressed to `24.2%`.

4. **Mixed 50/50 recovery resolved most of the tradeoff.**  
   It achieved static `k=10` ASR of `2.1%` and gen3 `k=20/40` ASR of `8.7%` and `13.0%`.

5. **The mixed model became harder to remine against under the current search budget.**  
   Only `5/590` candidates were kept in the gen4 hardcore stress remine.

6. **The mechanism appears to be stronger recovery/refusal pressure rather than simple distribution collapse.**  
   Mixed recovery increased refusal margin broadly, while KL-to-static remained large or increased after generation began.

## Limitations

- Most safety scores here use the keyword evaluator. The key behavioral comparisons should be validated with GPT-4 or another stronger judge before final publication.
- The gen3 hardcore set has only 23 prompts. It is excellent for paired mechanism analysis but too small for stable population-level ASR.
- The gen4 remine yielded only 5 kept prefixes, so its ASR is a stress-test signal, not a robust estimate.
- The trajectory analysis uses selected refusal/compliance token sets. Refusal margin is informative but not a complete distributional safety metric.
- The hidden-state subspace analysis uses a low-dimensional refusal direction/subspace and should not be read as ruling out other safety-relevant directions.

## Suggested Next Steps

1. Run GPT-4 judge on the key model outputs:
   - original augmented vs mixed on gen3 same23 `k=20/40`
   - adaptive-only vs mixed on static harmful prefixes `k=10/20/40`
   - the 5 gen4 kept prefixes

2. Repeat remine on a larger held-out set, not only the 23 hardcore prompts, if compute allows.

3. Add a utility evaluation for the mixed model to check whether the stronger refusal margin causes over-refusal.

4. Run the same paired trajectory analysis with `kl_reference_condition=refusal` to directly test whether mixed trajectories become closer to the refusal distribution over response steps.

