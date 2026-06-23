#!/usr/bin/env bash
set -euo pipefail

# Fine-tune a Qwen model on the llama2 safety dataset (same training data as
# the Llama-3.2-3B augmented model), producing a clean cross-architecture
# comparison without HEx-PHI train/eval leakage.
#
# Mirrors finetune_qwen_augmented_tinker.sh but fixes the safety data path
# to llama2_safety_data_direct.jsonl instead of Harmful-HEx-PHI.jsonl.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
RUN_NAME="${RUN_NAME:-qwen3_4b_llama2_safety_augmented}"
OUT_DIR="${OUT_DIR:-logs/qwen/${RUN_NAME}}"

SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-32}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
ALPHA="${ALPHA:-0.2}"
AUGMENT_PROBABILITY="${AUGMENT_PROBABILITY:-0.5}"
MAX_HARMFUL_PREFIX_TOKENS="${MAX_HARMFUL_PREFIX_TOKENS:-100}"
LOG_EVERY="${LOG_EVERY:-10}"

SAFETY_DATA_PATH="${SAFETY_DATA_PATH:-finetuning_buckets/datasets/data/tasks/data_augmentation/llama2_safety_data_direct.jsonl}"

mkdir -p "$OUT_DIR"

train_log_path="${OUT_DIR}/train.log"
checkpoint_dir="${OUT_DIR}/checkpoint"

echo "Training Qwen model on llama2 safety dataset"
echo "  model:       ${MODEL_NAME}"
echo "  output:      ${OUT_DIR}"
echo "  safety data: ${SAFETY_DATA_PATH}"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" finetune_tinker.py \
  --model_name "$MODEL_NAME" \
  --safety_data_path "$SAFETY_DATA_PATH" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --lora_rank "$LORA_RANK" \
  --max_length "$MAX_LENGTH" \
  --alpha "$ALPHA" \
  --augment_probability "$AUGMENT_PROBABILITY" \
  --max_harmful_prefix_tokens "$MAX_HARMFUL_PREFIX_TOKENS" \
  --log_every "$LOG_EVERY" \
  --log_path "$checkpoint_dir" \
  2>&1 | tee "$train_log_path"

"$PYTHON_BIN" - "$train_log_path" "${OUT_DIR}/checkpoint_paths.json" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
save_path = Path(sys.argv[2])
text = log_path.read_text()

match = re.search(r"Saved checkpoint paths:\n(\{.*?\})\s*$", text, re.S)
if not match:
    raise SystemExit(f"Could not parse checkpoint paths from {log_path}")

checkpoint_paths = json.loads(match.group(1))
save_path.write_text(json.dumps(checkpoint_paths, indent=2) + "\n")

sampler_path = (
    checkpoint_paths.get("sampler_path")
    or checkpoint_paths.get("sampler_weights")
    or checkpoint_paths.get("sampler")
)
if sampler_path:
    (save_path.parent / "sampler_path.txt").write_text(str(sampler_path) + "\n")
    print(f"sampler_path={sampler_path}")
else:
    print("No sampler path key found in checkpoint paths JSON.")
PY

echo "Done. Checkpoint metadata written to ${OUT_DIR}/checkpoint_paths.json"
echo "Sampler path written to ${OUT_DIR}/sampler_path.txt"

SAMPLER_PATH="$(<"${OUT_DIR}/sampler_path.txt")"

echo ""
echo "Evaluating on HEx-PHI static harmful prefix set"
echo "  model: ${SAMPLER_PATH}"

EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_KS="${EVAL_KS:-5 10 20 40}"

for k in $EVAL_KS; do
  echo "  k=${k}"
  "$PYTHON_BIN" eval_safety_tinker.py \
    --model_path "$SAMPLER_PATH" \
    --prompt_style qwen \
    --bench hex-phi_with_harmful_prefix \
    --eval_template plain \
    --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --evaluator key_word \
    --concurrency "$EVAL_CONCURRENCY" \
    --save_path "${OUT_DIR}/eval_hex_phi_static_k${k}.json"
done

echo "Done. Eval results written to ${OUT_DIR}/eval_hex_phi_static_k*.json"
