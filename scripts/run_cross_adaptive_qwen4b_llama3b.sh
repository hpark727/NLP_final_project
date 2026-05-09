#!/usr/bin/env bash
set -euo pipefail

# Evaluate adaptive prefixes mined by final Qwen4B and Llama3B students on both
# final target models. This produces a 2x2 transfer matrix:
#   target Qwen4B  x  {Qwen4B-mined, Llama3B-mined prefixes}
#   target Llama3B x  {Qwen4B-mined, Llama3B-mined prefixes}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

QWEN_RUN_DIR="${QWEN_RUN_DIR:-logs/teacher_forcing_qwen30b_to_qwen4b_zigzag}"
LLAMA_RUN_DIR="${LLAMA_RUN_DIR:-logs/teacher_forcing_qwen30b_to_llama3b_zigzag_v2}"
ITERATION="${ITERATION:-3}"

OUT_DIR="${OUT_DIR:-logs/cross_adaptive_qwen4b_llama3b}"
EVAL_KS="${EVAL_KS:-5 10 20 40 full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
CONCURRENCY="${CONCURRENCY:-16}"
EVALUATOR="${EVALUATOR:-key_word}"
EVAL_TEMPLATE="${EVAL_TEMPLATE:-plain}"
OUTPUT_DISTRIBUTION_STEPS="${OUTPUT_DISTRIBUTION_STEPS:-0}"
OUTPUT_DISTRIBUTION_TOPK="${OUTPUT_DISTRIBUTION_TOPK:-20}"
FORCE="${FORCE:-0}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-}"
QWEN_TARGET_LABEL="${QWEN_TARGET_LABEL:-qwen4b}"
LLAMA_TARGET_LABEL="${LLAMA_TARGET_LABEL:-llama3b}"
QWEN_PREFIX_PATH="${QWEN_PREFIX_PATH:-${QWEN_RUN_DIR}/iter_${ITERATION}/mined_student_iter_${ITERATION}.jsonl}"
LLAMA_PREFIX_PATH="${LLAMA_PREFIX_PATH:-${LLAMA_RUN_DIR}/iter_${ITERATION}/mined_student_iter_${ITERATION}.jsonl}"

read_path_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "Missing path file: ${path}" >&2
    exit 1
  fi
  head -n 1 "$path"
}

if [[ -z "$QWEN_MODEL_PATH" ]]; then
  QWEN_MODEL_PATH="$(read_path_file "${QWEN_RUN_DIR}/iter_${ITERATION}/student_sampler_path.txt")"
fi
if [[ -z "$LLAMA_MODEL_PATH" ]]; then
  LLAMA_MODEL_PATH="$(read_path_file "${LLAMA_RUN_DIR}/iter_${ITERATION}/student_sampler_path.txt")"
fi

for required_path in "$QWEN_PREFIX_PATH" "$LLAMA_PREFIX_PATH"; do
  if [[ ! -s "$required_path" ]]; then
    echo "Missing prefix file: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

run_eval() {
  local target_label="$1"
  local model_path="$2"
  local prompt_style="$3"
  local prefix_label="$4"
  local prefix_path="$5"
  local k="$6"

  local token_arg="$k"
  if [[ "$k" == "full" ]]; then
    token_arg="0"
  fi

  local save_path="${OUT_DIR}/${target_label}__${prefix_label}__k${k}.json"
  local distribution_dir="${OUT_DIR}/output_distribution/${target_label}__${prefix_label}__k${k}"
  if [[ "$FORCE" != "1" && -s "$save_path" ]]; then
    echo "Reuse ${save_path}"
    return
  fi

  mkdir -p "$distribution_dir"
  echo "Eval target=${target_label} prefix_source=${prefix_label} k=${k}"
  local args=(
    eval_safety_tinker.py
    --model_path "$model_path" \
    --prompt_style "$prompt_style" \
    --bench hex-phi_with_custom_prefix \
    --custom_prefix_path "$prefix_path" \
    --custom_prefix_tokens "$token_arg" \
    --eval_template "$EVAL_TEMPLATE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --save_output_distribution_dir "$distribution_dir" \
    --output_distribution_steps "$OUTPUT_DISTRIBUTION_STEPS" \
    --output_distribution_topk "$OUTPUT_DISTRIBUTION_TOPK" \
    --evaluator "$EVALUATOR" \
    --concurrency "$CONCURRENCY" \
    --save_path "$save_path"
  )
  if [[ -n "$MAX_EXAMPLES" ]]; then
    args+=(--max_examples "$MAX_EXAMPLES")
  fi
  "$PYTHON_BIN" "${args[@]}"
}

echo "Cross-adaptive Qwen4B/Llama3B eval"
echo "  qwen_target: ${QWEN_TARGET_LABEL}"
echo "  qwen_model: ${QWEN_MODEL_PATH}"
echo "  llama_target: ${LLAMA_TARGET_LABEL}"
echo "  llama_model: ${LLAMA_MODEL_PATH}"
echo "  qwen_prefixes: ${QWEN_PREFIX_PATH}"
echo "  llama_prefixes: ${LLAMA_PREFIX_PATH}"
echo "  out_dir: ${OUT_DIR}"
if [[ -n "$MAX_EXAMPLES" ]]; then
  echo "  max_examples: ${MAX_EXAMPLES}"
fi

for k in $EVAL_KS; do
  run_eval "$QWEN_TARGET_LABEL" "$QWEN_MODEL_PATH" qwen qwen4b_mined "$QWEN_PREFIX_PATH" "$k"
  run_eval "$QWEN_TARGET_LABEL" "$QWEN_MODEL_PATH" qwen llama3b_mined "$LLAMA_PREFIX_PATH" "$k"
  run_eval "$LLAMA_TARGET_LABEL" "$LLAMA_MODEL_PATH" llama3 llama3b_mined "$LLAMA_PREFIX_PATH" "$k"
  run_eval "$LLAMA_TARGET_LABEL" "$LLAMA_MODEL_PATH" llama3 qwen4b_mined "$QWEN_PREFIX_PATH" "$k"
done

"$PYTHON_BIN" analyze_cross_adaptive_eval.py --eval_dir "$OUT_DIR"

echo "Done."
echo "Summary: ${OUT_DIR}/cross_adaptive_summary.csv"
echo "Report: ${OUT_DIR}/cross_adaptive_summary.md"
