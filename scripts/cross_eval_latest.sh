#!/usr/bin/env bash
# Cross-eval: Llama-mined prefixes on Qwen 50/50, and Qwen-mined prefixes on Llama augmented.
# Uses BOS-clean Llama prefixes and most recent model checkpoints.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
CONCURRENCY="${EVAL_CONCURRENCY:-16}"
MAX_NEW_TOKENS=512

# Models
MODEL_LLAMA="tinker://de5327cc-9cda-57db-a585-4134fa998428:train:0/sampler_weights/final"
MODEL_QWEN="tinker://18499ced-0ff5-55f1-9d4e-3d8bd8e3ea67:train:0/sampler_weights/final"

# Prefix files (BOS-clean for Llama-mined; Qwen-mined has no BOS contamination)
PREFIX_LLAMA="logs/adaptive_prefixes/mined_augmented_tinker_bos_clean.jsonl"
PREFIX_QWEN="logs/qwen/qwen3_4b_llama2_safety_augmented/adaptive_mining/mined_adaptive_prefixes.jsonl"

OUT_DIR="logs/cross_eval_latest"
mkdir -p "$OUT_DIR"

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

# ── 1. Llama-mined prefixes → Qwen 50/50 (cross-model transfer) ──────────────
echo "=== Llama-mined prefixes on Qwen 50/50 ==="
for k in 5 10 20 40; do
  echo "  k=$k"
  run_eval "$MODEL_QWEN" qwen "$PREFIX_LLAMA" "$k" \
    "${OUT_DIR}/qwen_on_llama_mined_k${k}.json"
done

# ── 2. Qwen-mined prefixes → Llama augmented (cross-model transfer) ──────────
echo "=== Qwen-mined prefixes on Llama augmented ==="
for k in 5 10 20 40; do
  echo "  k=$k"
  run_eval "$MODEL_LLAMA" llama3 "$PREFIX_QWEN" "$k" \
    "${OUT_DIR}/llama_on_qwen_mined_k${k}.json"
done

echo ""
echo "Cross-eval complete. Results in ${OUT_DIR}/"
