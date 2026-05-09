#!/usr/bin/env bash
set -euo pipefail

# Mine adaptive prefixes against the original static-augmented Llama3B checkpoint
# using the same mining constraints as the Qwen static-augmented adaptive run:
#
#   logs/qwen/qwen3_4b_static_augmented/adaptive_mining/mined_adaptive_prefixes.metadata.json
#
# This is intended to produce a protocol-matched Llama prefix set for controlled
# Qwen/Llama cross-adaptive evaluation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

RUN_NAME="${RUN_NAME:-llama3b_static_augmented_qwen_matched}"
OUT_DIR="${OUT_DIR:-logs/adaptive_prefixes/${RUN_NAME}/adaptive_mining}"

# Original static-augmented Llama checkpoint used in prior controlled runs.
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-tinker://de5327cc-9cda-57db-a585-4134fa998428:train:0/sampler_weights/final}"
PROMPT_STYLE="${PROMPT_STYLE:-llama3}"

# Qwen-matched mining constraints.
MAX_EXAMPLES="${MAX_EXAMPLES:-330}"
SEED_PREFIX_TOKENS="${SEED_PREFIX_TOKENS:-10,20}"
SAMPLES_PER_SEED="${SAMPLES_PER_SEED:-3}"
MAX_KEPT_PER_PROMPT="${MAX_KEPT_PER_PROMPT:-2}"
CONTINUATION_TOKENS="${CONTINUATION_TOKENS:-20}"
VERIFICATION_TOKENS="${VERIFICATION_TOKENS:-128}"
MAX_PREFIX_TOKENS="${MAX_PREFIX_TOKENS:-80}"
MINING_TEMPERATURE="${MINING_TEMPERATURE:-1.0}"
MINING_TOP_P="${MINING_TOP_P:-0.95}"
VERIFICATION_TEMPERATURE="${VERIFICATION_TEMPERATURE:-0.7}"
VERIFICATION_TOP_P="${VERIFICATION_TOP_P:-0.9}"
MINING_CONCURRENCY="${MINING_CONCURRENCY:-16}"
RANDOM_SEED="${RANDOM_SEED:-0}"
EVALUATOR="${EVALUATOR:-key_word}"

# Optional post-mining eval on the same Llama checkpoint.
EVAL_KS="${EVAL_KS:-5 10 20 40 full}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.9}"
EVAL_TOP_P="${EVAL_TOP_P:-0.6}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EVAL_OUTPUT_DISTRIBUTION_STEPS="${EVAL_OUTPUT_DISTRIBUTION_STEPS:-0}"
EVAL_OUTPUT_DISTRIBUTION_TOPK="${EVAL_OUTPUT_DISTRIBUTION_TOPK:-20}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "$OUT_DIR"

mine_path="${OUT_DIR}/mined_adaptive_prefixes.jsonl"
all_candidates_path="${OUT_DIR}/all_candidates_adaptive_prefixes.jsonl"
metadata_path="${OUT_DIR}/mined_adaptive_prefixes.metadata.json"

echo "Mining Llama adaptive prefixes with Qwen-matched constraints"
echo "  model_path: ${LLAMA_MODEL_PATH}"
echo "  prompt_style: ${PROMPT_STYLE}"
echo "  seed_prefix_tokens: ${SEED_PREFIX_TOKENS}"
echo "  samples_per_seed: ${SAMPLES_PER_SEED}"
echo "  max_kept_per_prompt: ${MAX_KEPT_PER_PROMPT}"
echo "  verification_tokens: ${VERIFICATION_TOKENS}"
echo "  output: ${OUT_DIR}"

"$PYTHON_BIN" mine_adaptive_prefixes_tinker.py \
  --model_path "$LLAMA_MODEL_PATH" \
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
  --evaluator "$EVALUATOR" \
  --save_path "$mine_path" \
  --metadata_path "$metadata_path" \
  --save_all_candidates_path "$all_candidates_path"

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "Evaluating matched mined prefixes on the same Llama checkpoint"
  for k in $EVAL_KS; do
    if [[ "$k" == "full" ]]; then
      custom_prefix_tokens=0
      suffix="full"
    else
      custom_prefix_tokens="$k"
      suffix="$k"
    fi

    echo "  k=${k}"
    "$PYTHON_BIN" eval_safety_tinker.py \
      --model_path "$LLAMA_MODEL_PATH" \
      --prompt_style "$PROMPT_STYLE" \
      --bench hex-phi_with_custom_prefix \
      --custom_prefix_path "$mine_path" \
      --custom_prefix_tokens "$custom_prefix_tokens" \
      --eval_template plain \
      --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
      --temperature "$EVAL_TEMPERATURE" \
      --top_p "$EVAL_TOP_P" \
      --output_distribution_steps "$EVAL_OUTPUT_DISTRIBUTION_STEPS" \
      --output_distribution_topk "$EVAL_OUTPUT_DISTRIBUTION_TOPK" \
      --evaluator "$EVALUATOR" \
      --concurrency "$EVAL_CONCURRENCY" \
      --save_path "${OUT_DIR}/eval_llama_augmented_on_qwen_matched_mined_k${suffix}.json"
  done
fi

echo "Done."
echo "Mined prefixes: ${mine_path}"
echo "All candidates: ${all_candidates_path}"
echo "Metadata: ${metadata_path}"
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "Evaluations: ${OUT_DIR}/eval_llama_augmented_on_qwen_matched_mined_k*.json"
fi
