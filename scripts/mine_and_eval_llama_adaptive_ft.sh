#!/usr/bin/env bash
set -euo pipefail

# Mine adaptive prefixes from the adaptive-recovery Llama model against the
# full HEx-PHI set (330 examples), then evaluate that model on the mined
# prefix set at k=5/10/20/40.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

MODEL_PATH="${MODEL_PATH:-tinker://9c0f55da-6b2a-5887-a5e2-85f62b10a661:train:0/sampler_weights/final}"
OUT_DIR="${OUT_DIR:-logs/adaptive_prefixes/llama_safety_dataset/llama_adaptive_ft/adaptive_mining}"

PROMPT_STYLE="llama3"
MAX_EXAMPLES="330"
SEED_PREFIX_TOKENS="5,10,20"
SAMPLES_PER_SEED="3"
MAX_KEPT_PER_PROMPT="1"
CONTINUATION_TOKENS="20"
VERIFICATION_TOKENS="256"
MAX_PREFIX_TOKENS="80"
MINING_TEMPERATURE="1.0"
MINING_TOP_P="0.95"
VERIFICATION_TEMPERATURE="0.7"
VERIFICATION_TOP_P="0.9"
MINING_CONCURRENCY="${MINING_CONCURRENCY:-16}"
RANDOM_SEED="0"

EVAL_MAX_NEW_TOKENS="512"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EVAL_EVALUATOR="key_word"
EVAL_KS="5 10 20 40"

mkdir -p "$OUT_DIR"

mine_path="${OUT_DIR}/mined_adaptive_prefixes.jsonl"
all_candidates_path="${OUT_DIR}/all_candidates_adaptive_prefixes.jsonl"
metadata_path="${OUT_DIR}/mined_adaptive_prefixes.metadata.json"

echo "Mining adaptive prefixes from adaptive-recovery Llama model"
echo "  model_path: ${MODEL_PATH}"
echo "  output: ${OUT_DIR}"

"$PYTHON_BIN" mine_adaptive_prefixes_tinker.py \
  --model_path "$MODEL_PATH" \
  --prompt_style "$PROMPT_STYLE" \
  --eval_template null \
  --max_examples "$MAX_EXAMPLES" \
  --seed_prefix_tokens "$SEED_PREFIX_TOKENS" \
  --samples_per_seed "$SAMPLES_PER_SEED" \
  --max_kept_per_prompt "$MAX_KEPT_PER_PROMPT" \
  --continuation_tokens "$CONTINUATION_TOKENS" \
  --verification_tokens "$VERIFICATION_TOKENS" \
  --max_prefix_tokens "$MAX_PREFIX_TOKENS" \
  --mining_temperature "$MINING_TEMPERATURE" \
  --mining_top_p "$MINING_TOP_P" \
  --verification_temperature "$VERIFICATION_TEMPERATURE" \
  --verification_top_p "$VERIFICATION_TOP_P" \
  --random_seed "$RANDOM_SEED" \
  --concurrency "$MINING_CONCURRENCY" \
  --evaluator key_word \
  --save_path "$mine_path" \
  --metadata_path "$metadata_path" \
  --save_all_candidates_path "$all_candidates_path"

echo "Evaluating adaptive-recovery model on its own mined prefixes"
for k in $EVAL_KS; do
  echo "  k=${k}"
  "$PYTHON_BIN" eval_safety_tinker.py \
    --model_path "$MODEL_PATH" \
    --prompt_style "$PROMPT_STYLE" \
    --bench hex-phi_with_custom_prefix \
    --custom_prefix_path "$mine_path" \
    --custom_prefix_tokens "$k" \
    --eval_template plain \
    --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --evaluator "$EVAL_EVALUATOR" \
    --concurrency "$EVAL_CONCURRENCY" \
    --save_path "${OUT_DIR}/eval_adaptive_ft_on_self_mined_k${k}.json"
done

echo "Done."
echo "Mined prefixes : ${mine_path}"
echo "All candidates : ${all_candidates_path}"
echo "Evaluations    : ${OUT_DIR}/eval_adaptive_ft_on_self_mined_k*.json"
