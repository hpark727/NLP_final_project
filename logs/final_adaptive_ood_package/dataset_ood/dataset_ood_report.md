# Dataset OOD Classifier: Aegis vs HEx-PHI

## Question

Is Aegis meaningfully out-of-distribution relative to HEx-PHI, or is it just another unsafe prompt set?

## Setup

- HEx-PHI examples: `330`
- Aegis unsafe prompt + unsafe response examples sampled for balance: `330` from the prepared `637`
- Label: `Aegis=1`, `HEx-PHI=0`
- Evaluation: 5-fold stratified cross-validation

## Classifier Results

| Model | Input | Accuracy | AUC |
|---|---|---:|---:|
| `prompt_tfidf` | `prompt_text` | 0.882 | 0.949 |
| `prompt_plus_prefix_tfidf` | `combined_text` | 0.971 | 0.995 |
| `surface_features` | `surface` | 0.945 | 0.984 |


![Dataset OOD scores](figures/dataset_ood_score_distributions.png)

## Interpretable Surface Differences

| Feature | HEx-PHI mean | Aegis mean | Standardized Aegis-HEx |
|---|---:|---:|---:|
| `prompt_char_len` | 164.839 | 72.206 | -1.17 |
| `char_entropy` | 4.480 | 4.305 | -1.10 |
| `prefix_line_count` | 15.179 | 1.176 | -1.04 |
| `prompt_token_len` | 27.048 | 14.439 | -1.02 |
| `prompt_question_marks` | 0.288 | 0.736 | 0.84 |
| `instructional_count` | 2.097 | 0.342 | -0.68 |
| `combined_char_len` | 1103.985 | 704.830 | -0.61 |
| `first_person_count` | 1.694 | 3.858 | 0.54 |


![Dataset OOD surface deltas](figures/dataset_ood_surface_deltas.png)

## Interpretation

If a lightweight classifier can distinguish Aegis from HEx-PHI with high cross-validated AUC, then Aegis is a concrete prompt-source OOD check rather than a cosmetic extra benchmark. The prompt-only model is especially important: it tests whether the user requests themselves differ, not merely the generated unsafe prefixes.

This does not mean Aegis is "harder" in a universal sense. It means it is distributionally different. The Aegis ASR results should therefore be read as a cross-dataset stress test, not as a direct continuation of the HEx-PHI adaptive-prefix benchmark.
