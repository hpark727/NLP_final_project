#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
CONCURRENCY="${EVAL_CONCURRENCY:-16}"
MAX_NEW_TOKENS=512

run_eval() {
  local model_path=$1
  local prompt_style=$2
  local prefix_file=$3
  local k=$4
  local save_path=$5
  "$PYTHON_BIN" eval_safety_tinker.py \
    --model_path "$model_path" \
    --prompt_style "$prompt_style" \
    --bench hex-phi_with_custom_prefix \
    --custom_prefix_path "$prefix_file" \
    --custom_prefix_tokens "$k" \
    --eval_template plain \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --evaluator key_word \
    --concurrency "$CONCURRENCY" \
    --save_path "$save_path"
}

# ── 1. Llama augmented (de5327cc) on gen2_v2 clean ──────────────────────────
MODEL_DE="tinker://de5327cc-9cda-57db-a585-4134fa998428:train:0/sampler_weights/final"
PREFIX_GEN2="logs/adaptive_prefixes/mined_augmented_tinker_gen2_v2_bos_clean.jsonl"
OUT_DE="logs/adaptive_prefixes"
echo "=== Llama augmented on gen2_v2 (BOS clean) ==="
for k in 5 10 20 40; do
  echo "  k=$k"
  run_eval "$MODEL_DE" llama3 "$PREFIX_GEN2" "$k" \
    "${OUT_DE}/eval_augmented_on_gen2_v2_bos_clean_k${k}.json"
done

# ── 2. Llama 50/50 (e8c79993) on mined_augmented_tinker clean ───────────────
MODEL_5050="tinker://e8c79993-e9c7-564c-8324-bf56d9d22b79:train:0/sampler_weights/final"
PREFIX_TINKER="logs/adaptive_prefixes/mined_augmented_tinker_bos_clean.jsonl"
OUT_5050="logs/adaptive_prefixes/llama_safety_dataset/llama_50_50_fixed_compute_ft"
echo "=== Llama 50/50 on mined_augmented_tinker (BOS clean) ==="
for k in 5 10 20 40; do
  echo "  k=$k"
  run_eval "$MODEL_5050" llama3 "$PREFIX_TINKER" "$k" \
    "${OUT_5050}/eval_adaptive_bos_clean_k${k}.json"
done

# ── 3. Llama adaptive recovery (9c0f55da) on mined_augmented_tinker clean ───
MODEL_AR="tinker://9c0f55da-6b2a-5887-a5e2-85f62b10a661:train:0/sampler_weights/final"
OUT_AR="logs/adaptive_prefixes/llama_safety_dataset/llama_adaptive_ft"
echo "=== Llama adaptive recovery on mined_augmented_tinker (BOS clean) ==="
for k in 5 10 20 40; do
  echo "  k=$k"
  run_eval "$MODEL_AR" llama3 "$PREFIX_TINKER" "$k" \
    "${OUT_AR}/eval_adaptive_bos_clean_k${k}.json"
done

# ── 4. Llama adaptive recovery on its own self-mined prefixes (clean) ────────
PREFIX_SELF="logs/adaptive_prefixes/llama_safety_dataset/llama_adaptive_ft/adaptive_mining/mined_adaptive_prefixes_bos_clean.jsonl"
OUT_AR_SELF="${OUT_AR}/adaptive_mining"
echo "=== Llama adaptive recovery on self-mined (BOS clean) ==="
for k in 5 10 20 40; do
  echo "  k=$k"
  run_eval "$MODEL_AR" llama3 "$PREFIX_SELF" "$k" \
    "${OUT_AR_SELF}/eval_adaptive_ft_on_self_mined_bos_clean_k${k}.json"
done

# ── 5. Llama augmented (de5327cc) on qwen_matched clean ─────────────────────
PREFIX_QMATCH="logs/adaptive_prefixes/llama3b_static_augmented_qwen_matched/adaptive_mining/mined_adaptive_prefixes_bos_clean.jsonl"
OUT_QMATCH="logs/adaptive_prefixes/llama3b_static_augmented_qwen_matched/adaptive_mining"
echo "=== Llama augmented on qwen_matched (BOS clean) ==="
for k in 5 10 20 40; do
  echo "  k=$k"
  run_eval "$MODEL_DE" llama3 "$PREFIX_QMATCH" "$k" \
    "${OUT_QMATCH}/eval_llama_augmented_on_qwen_matched_bos_clean_k${k}.json"
done

# ── 6. Qwen 50/50 (18ee75b2) on mined_augmented_tinker clean ────────────────
MODEL_Q5050="tinker://18ee75b2-c4ed-518c-b6d5-d692a0051fac:train:0/sampler_weights/final"
OUT_Q5050="logs/qwen/qwen3_4b_llama_safety_50_50_fixed_compute"
echo "=== Qwen 50/50 on mined_augmented_tinker (BOS clean) ==="
for k in 5 10 20 40; do
  echo "  k=$k"
  run_eval "$MODEL_Q5050" qwen "$PREFIX_TINKER" "$k" \
    "${OUT_Q5050}/eval_adaptive_bos_clean_k${k}.json"
done

echo ""
echo "All BOS-clean re-evals complete."
