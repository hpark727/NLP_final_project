#!/usr/bin/env bash
set -euo pipefail

# Fine-tune a Qwen static-prefill-augmented safety model with Tinker.
#
# This reproduces the original "prefill-resistant SFT" setup for Qwen:
#   harmful prompt -> harmful assistant prefix -> refusal continuation
#
# By default, finetune_tinker.py samples a random harmful assistant prefix from
# each row's `harmful` field because --augmentation_prefix_field is intentionally
# omitted here.

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
RUN_NAME="${RUN_NAME:-qwen3_4b_static_augmented}"
OUT_DIR="${OUT_DIR:-logs/qwen/${RUN_NAME}}"

SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
ALPHA="${ALPHA:-0.2}"
AUGMENT_PROBABILITY="${AUGMENT_PROBABILITY:-0.5}"
MAX_HARMFUL_PREFIX_TOKENS="${MAX_HARMFUL_PREFIX_TOKENS:-100}"
LOG_EVERY="${LOG_EVERY:-10}"

SAFETY_DATA_PATH="${SAFETY_DATA_PATH:-finetuning_buckets/datasets/data/tasks/data_augmentation/llama2_safety_data_direct.jsonl}"

mkdir -p "$OUT_DIR"

train_log_path="${OUT_DIR}/train.log"
checkpoint_dir="${OUT_DIR}/checkpoint"

echo "Training Qwen augmented model"
echo "  model: ${MODEL_NAME}"
echo "  output: ${OUT_DIR}"
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
echo "Sampler path, if found, written to ${OUT_DIR}/sampler_path.txt"
