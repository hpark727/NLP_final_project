#!/usr/bin/env zsh
set -euo pipefail

set -a
source .env
set +a

PREFIX_PATH="${PREFIX_PATH:-logs/final_adaptive_ood_package/aegis/aegis_unsafe_prefixes_500.jsonl}"
OUT_ROOT="${OUT_ROOT:-logs/final_adaptive_ood_package/aegis_eval}"
mkdir -p "$OUT_ROOT"

typeset -A MODELS
MODELS[original_augmented]="tinker://de5327cc-9cda-57db-a585-4134fa998428:train:0/sampler_weights/final"
MODELS[adaptive_only_recovery]="tinker://d82c30b2-da1f-597f-9d58-b72f5ecef244:train:0/sampler_weights/final"
MODELS[mixed_50_50_recovery]="tinker://9e8229e1-3d61-58d0-99ab-f04314679ac2:train:0/sampler_weights/final"

for model_name in original_augmented adaptive_only_recovery mixed_50_50_recovery; do
  for k in 5 10 20 40; do
    echo
    echo "### Aegis OOD eval: $model_name k=$k"
    ./.venv/bin/python eval_safety_tinker.py \
      --model_path "${MODELS[$model_name]}" \
      --prompt_style llama3 \
      --bench hex-phi_with_custom_prefix \
      --eval_template null \
      --custom_prefix_path "$PREFIX_PATH" \
      --custom_prefix_tokens "$k" \
      --max_new_tokens 512 \
      --temperature 0.9 \
      --top_p 0.6 \
      --concurrency 16 \
      --output_distribution_steps 0 \
      --evaluator key_word \
      --save_path "$OUT_ROOT/${model_name}_aegis_k${k}.json"
  done
done
