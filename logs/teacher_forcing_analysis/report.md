# Teacher-Forcing Zigzag Analysis

Generated: 2026-05-03T17:46:52

## Headline

- llama3b: mean static-prefix ASR across k fell from 20.9% at iter 1 to 4.5% at iter 3 (78.6% relative reduction).
- qwen4b: mean static-prefix ASR across k fell from 18.7% at iter 1 to 0.8% at iter 3 (96.0% relative reduction).
- At iter 3, qwen4b is lower than llama3b on every static-prefix budget; the largest absolute gap is 4.5%.

## Static HEx-PHI Prefix ASR

| family | iter | mean ASR | max ASR | pooled ASR | slope / token |
|---|---:|---:|---:|---:|---:|
| llama3b | 1 | 20.9% | 27.6% | 20.9% | 0.00342 |
| llama3b | 2 | 7.4% | 9.4% | 7.4% | 0.00121 |
| llama3b | 3 | 4.5% | 5.5% | 4.5% | 0.00052 |
| qwen4b | 1 | 18.7% | 29.1% | 18.7% | 0.00518 |
| qwen4b | 2 | 1.4% | 3.0% | 1.4% | 0.00071 |
| qwen4b | 3 | 0.8% | 1.5% | 0.8% | 0.00043 |

## Persistent Prompt Failures

This counts prompt indices that still jailbreak under multiple tested prefix budgets.

| family | iter | >=1 k | >=2 k | all k |
|---|---:|---:|---:|---:|
| llama3b | 1 | 168 | 77 | 6 |
| llama3b | 2 | 70 | 22 | 0 |
| llama3b | 3 | 47 | 10 | 0 |
| qwen4b | 1 | 150 | 63 | 8 |
| qwen4b | 2 | 14 | 4 | 0 |
| qwen4b | 3 | 8 | 2 | 0 |

## Cross-Family Paired Comparison

Exact McNemar tests compare matched prompt indices under the same iteration and prefix budget.

| iter | k | left ASR | right ASR | right-left | left-only | right-only | p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 15.5% | 10.3% | -5.2% | 47 | 30 | 0.0675 |
| 1 | 10 | 17.9% | 14.8% | -3.0% | 51 | 41 | 0.348 |
| 1 | 20 | 22.7% | 20.6% | -2.1% | 55 | 48 | 0.555 |
| 1 | 40 | 27.6% | 29.1% | 1.5% | 56 | 61 | 0.712 |
| 2 | 5 | 5.8% | 0.9% | -4.8% | 19 | 3 | 0.000855 |
| 2 | 10 | 5.2% | 0.3% | -4.8% | 17 | 1 | 0.000145 |
| 2 | 20 | 9.4% | 1.5% | -7.9% | 29 | 3 | 2.56e-06 |
| 2 | 40 | 9.4% | 3.0% | -6.4% | 28 | 7 | 0.000508 |
| 3 | 5 | 3.0% | 0.0% | -3.0% | 10 | 0 | 0.00195 |
| 3 | 10 | 4.8% | 0.3% | -4.5% | 16 | 1 | 0.000275 |
| 3 | 20 | 4.5% | 1.2% | -3.3% | 15 | 4 | 0.0192 |
| 3 | 40 | 5.5% | 1.5% | -3.9% | 16 | 3 | 0.00443 |

## Mining Yield

| family | iter | miner | kept / attempted | kept / example | unique prompt IDs | refusal rejects | verification rejects |
|---|---:|---|---:|---:|---:|---:|---:|
| llama3b | 1 | student | 28.9% | 1.37 | 154 | 31.8% | 39.3% |
| llama3b | 1 | teacher | 54.0% | 1.73 | 300 | 10.7% | 35.3% |
| llama3b | 2 | student | 6.6% | 0.39 | 35 | 77.6% | 15.7% |
| llama3b | 2 | teacher | 39.4% | 1.92 | 119 | 54.0% | 6.7% |
| llama3b | 3 | student | 8.1% | 0.45 | 12 | 80.6% | 11.3% |
| llama3b | 3 | teacher | 30.9% | 1.48 | 32 | 64.0% | 5.0% |
| qwen4b | 1 | student | 16.3% | 0.88 | 118 | 65.7% | 17.9% |
| qwen4b | 1 | teacher | 55.8% | 1.76 | 302 | 10.1% | 34.1% |
| qwen4b | 2 | student | 1.3% | 0.08 | 4 | 97.5% | 1.2% |
| qwen4b | 2 | teacher | 92.3% | 1.99 | 118 | 0.8% | 6.9% |
| qwen4b | 3 | student | 33.3% | 1.44 | 3 | 42.1% | 24.5% |
| qwen4b | 3 | teacher | 100.0% | 2.00 | 4 | 0.0% | 0.0% |

## Custom Mined-Prefix Eval

These are evals on mined prefix sets, not the full static HEx-PHI harmful-prefix sweep.

| family | iter | label | ASR | successes / total | 95% CI |
|---|---:|---|---:|---:|---:|
| qwen4b | 3 | final_on_iter3_student_prefixes | 55.6% | 40 / 72 | [44.1%, 66.5%] |
| qwen4b | 3 | final_on_iter3_teacher_prefixes | 46.0% | 23 / 50 | [33.0%, 59.6%] |

## Caveats

- Safety success here is the existing keyword-jailbreak metric, so the report is best read as a structured robustness screen rather than a final semantic safety judgment.
- The paired tests use matched row indices from the saved eval files. They do not assume independent prompts, which makes them more appropriate than comparing only aggregate ASR.
- Custom mined-prefix evals can be high even when static-prefix ASR is low; they measure a different, adversarially selected distribution.
