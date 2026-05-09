#!/usr/bin/env bash
set -euo pipefail

# Mine adaptive prefixes against a Qwen augmented Tinker checkpoint, then
# evaluate that mined prefix set at k=5/10/20/40.
#
# Default AUGMENTED_MODEL_PATH_FILE points to the output of:
#   scripts/finetune_qwen_augmented_tinker.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

RUN_NAME="${RUN_NAME:-qwen3_4b_static_augmented}"
OUT_DIR="${OUT_DIR:-logs/qwen/${RUN_NAME}/adaptive_mining}"

AUGMENTED_MODEL_PATH="${AUGMENTED_MODEL_PATH:-}"
AUGMENTED_MODEL_PATH_FILE="${AUGMENTED_MODEL_PATH_FILE:-logs/qwen/${RUN_NAME}/sampler_path.txt}"
PROMPT_STYLE="${PROMPT_STYLE:-qwen}"

MAX_EXAMPLES="${MAX_EXAMPLES:-330}"
SEED_PREFIX_TOKENS="${SEED_PREFIX_TOKENS:-5,10,20}"
SAMPLES_PER_SEED="${SAMPLES_PER_SEED:-10}"
MAX_KEPT_PER_PROMPT="${MAX_KEPT_PER_PROMPT:-0}"
CONTINUATION_TOKENS="${CONTINUATION_TOKENS:-20}"
VERIFICATION_TOKENS="${VERIFICATION_TOKENS:-256}"
MAX_PREFIX_TOKENS="${MAX_PREFIX_TOKENS:-80}"
MINING_TEMPERATURE="${MINING_TEMPERATURE:-1.0}"
MINING_TOP_P="${MINING_TOP_P:-0.95}"
VERIFICATION_TEMPERATURE="${VERIFICATION_TEMPERATURE:-0.7}"
VERIFICATION_TOP_P="${VERIFICATION_TOP_P:-0.9}"
MINING_CONCURRENCY="${MINING_CONCURRENCY:-16}"
RANDOM_SEED="${RANDOM_SEED:-0}"

EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EVAL_OUTPUT_DISTRIBUTION_STEPS="${EVAL_OUTPUT_DISTRIBUTION_STEPS:-0}"
EVAL_OUTPUT_DISTRIBUTION_TOPK="${EVAL_OUTPUT_DISTRIBUTION_TOPK:-20}"
EVAL_EVALUATOR="${EVAL_EVALUATOR:-key_word}"
EVAL_KS="${EVAL_KS:-5 10 20 40}"

mkdir -p "$OUT_DIR"

if [[ -z "$AUGMENTED_MODEL_PATH" ]]; then
  if [[ ! -f "$AUGMENTED_MODEL_PATH_FILE" ]]; then
    echo "Missing augmented model path. Set AUGMENTED_MODEL_PATH or create ${AUGMENTED_MODEL_PATH_FILE}." >&2
    exit 1
  fi
  AUGMENTED_MODEL_PATH="$(<"$AUGMENTED_MODEL_PATH_FILE")"
fi

mine_path="${OUT_DIR}/mined_adaptive_prefixes.jsonl"
all_candidates_path="${OUT_DIR}/all_candidates_adaptive_prefixes.jsonl"
metadata_path="${OUT_DIR}/mined_adaptive_prefixes.metadata.json"

echo "Mining adaptive prefixes from augmented model"
echo "  model_path: ${AUGMENTED_MODEL_PATH}"
echo "  samples_per_seed: ${SAMPLES_PER_SEED}"
echo "  max_kept_per_prompt: ${MAX_KEPT_PER_PROMPT} (<=0 means keep all successes)"
echo "  output: ${OUT_DIR}"

"$PYTHON_BIN" mine_adaptive_prefixes_tinker.py \
  --model_path "$AUGMENTED_MODEL_PATH" \
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

echo "Evaluating mined adaptive prefixes"
for k in $EVAL_KS; do
  echo "  k=${k}"
  "$PYTHON_BIN" eval_safety_tinker.py \
    --model_path "$AUGMENTED_MODEL_PATH" \
    --prompt_style "$PROMPT_STYLE" \
    --bench hex-phi_with_custom_prefix \
    --custom_prefix_path "$mine_path" \
    --custom_prefix_tokens "$k" \
    --eval_template plain \
    --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --output_distribution_steps "$EVAL_OUTPUT_DISTRIBUTION_STEPS" \
    --output_distribution_topk "$EVAL_OUTPUT_DISTRIBUTION_TOPK" \
    --evaluator "$EVAL_EVALUATOR" \
    --concurrency "$EVAL_CONCURRENCY" \
    --save_path "${OUT_DIR}/eval_augmented_on_mined_adaptive_k${k}.json"
done

echo "Done."
echo "Mined prefixes: ${mine_path}"
echo "All candidates: ${all_candidates_path}"
echo "Metadata: ${metadata_path}"
echo "Evaluations: ${OUT_DIR}/eval_augmented_on_mined_adaptive_k*.json"
