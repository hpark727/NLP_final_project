#!/usr/bin/env bash
set -euo pipefail

# Download/register a finished Tinker Qwen sampler checkpoint as a local PEFT
# adapter under ckpts/, matching the layout used by the local Llama adapters.
#
# Typical use after scripts/finetune_qwen_augmented_tinker.sh finishes:
#
#   ./scripts/register_qwen_augmented_ckpt.sh
#
# Useful overrides:
#
#   RUN_NAME=qwen3_4b_static_augmented ./scripts/register_qwen_augmented_ckpt.sh
#   SAMPLER_PATH='tinker://...' ./scripts/register_qwen_augmented_ckpt.sh
#   ADAPTER_DIR=/path/to/downloaded/adapter ./scripts/register_qwen_augmented_ckpt.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
TINKER_BIN="${TINKER_BIN:-./.venv/bin/tinker}"

RUN_NAME="${RUN_NAME:-qwen3_4b_static_augmented}"
OUT_DIR="${OUT_DIR:-logs/qwen/${RUN_NAME}}"
CHECKPOINT_JSON="${CHECKPOINT_JSON:-${OUT_DIR}/checkpoint_paths.json}"
SAMPLER_PATH_FILE="${SAMPLER_PATH_FILE:-${OUT_DIR}/sampler_path.txt}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-ckpts/Qwen3-4B-Instruct-2507}"
CKPT_PARENT="${CKPT_PARENT:-ckpts/Qwen3-4B-Instruct-2507-Augmented}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
MAKE_LATEST_SYMLINK="${MAKE_LATEST_SYMLINK:-1}"

mkdir -p "$CKPT_PARENT"

if [[ ! -f "${BASE_MODEL_PATH}/config.json" ]]; then
  echo "Missing local Qwen base model at ${BASE_MODEL_PATH}/config.json" >&2
  echo "Download Qwen first, or set BASE_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507." >&2
  exit 1
fi

if [[ -n "${ADAPTER_DIR:-}" ]]; then
  if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    echo "ADAPTER_DIR does not contain adapter_config.json: ${ADAPTER_DIR}" >&2
    exit 1
  fi
  adapter_dir="${ADAPTER_DIR}"
else
  if [[ -z "${SAMPLER_PATH:-}" ]]; then
    if [[ -f "$SAMPLER_PATH_FILE" ]]; then
      SAMPLER_PATH="$(tr -d '[:space:]' < "$SAMPLER_PATH_FILE")"
    elif [[ -f "$CHECKPOINT_JSON" ]]; then
      SAMPLER_PATH="$("$PYTHON_BIN" - "$CHECKPOINT_JSON" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
for key in ("sampler_path", "sampler_weights", "sampler"):
    value = data.get(key)
    if value:
        print(value)
        break
else:
    raise SystemExit(f"No sampler path found in {sys.argv[1]}")
PY
)"
    else
      echo "No SAMPLER_PATH provided, and no checkpoint metadata found at:" >&2
      echo "  ${SAMPLER_PATH_FILE}" >&2
      echo "  ${CHECKPOINT_JSON}" >&2
      exit 1
    fi
  fi

  before_list="$(mktemp)"
  after_list="$(mktemp)"
  trap 'rm -f "$before_list" "$after_list"' EXIT

  find "$CKPT_PARENT" -maxdepth 1 -mindepth 1 -type d -print | sort > "$before_list"

  download_args=(checkpoint download "$SAMPLER_PATH" --output "$CKPT_PARENT")
  if [[ "$FORCE_DOWNLOAD" == "1" ]]; then
    download_args+=(--force)
  fi

  echo "Downloading sampler checkpoint:"
  echo "  ${SAMPLER_PATH}"
  "$TINKER_BIN" "${download_args[@]}"

  find "$CKPT_PARENT" -maxdepth 1 -mindepth 1 -type d -print | sort > "$after_list"
  adapter_dir="$(comm -13 "$before_list" "$after_list" | while read -r path; do
    if [[ -f "${path}/adapter_config.json" ]]; then
      echo "$path"
      break
    fi
  done)"

  if [[ -z "$adapter_dir" ]]; then
    adapter_dir="$(find "$CKPT_PARENT" -maxdepth 2 -name adapter_config.json -print \
      | xargs -n1 dirname \
      | sort \
      | tail -1)"
  fi

  if [[ -z "$adapter_dir" || ! -f "${adapter_dir}/adapter_config.json" ]]; then
    echo "Downloaded checkpoint, but could not find adapter_config.json under ${CKPT_PARENT}." >&2
    exit 1
  fi

  printf "%s\n" "$SAMPLER_PATH" > "${adapter_dir}/sampler_path.txt"
  mkdir -p "$OUT_DIR"
  printf "%s\n" "$SAMPLER_PATH" > "$SAMPLER_PATH_FILE"
fi

"$PYTHON_BIN" - "$adapter_dir" "$BASE_MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

adapter_dir = Path(sys.argv[1])
base_model_path = sys.argv[2]
config_path = adapter_dir / "adapter_config.json"

config = json.loads(config_path.read_text())
config["base_model_name_or_path"] = base_model_path
config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
(adapter_dir / "base_model_path.txt").write_text(base_model_path + "\n")
(adapter_dir / "checkpoint_complete").touch()
PY

if [[ "$MAKE_LATEST_SYMLINK" == "1" ]]; then
  ln -sfn "$(basename "$adapter_dir")" "${CKPT_PARENT}/latest"
fi

echo "Registered local Qwen adapter:"
echo "  ${adapter_dir}"
echo
echo "Use it with:"
echo "  --model_name_or_path \"${adapter_dir}\" --model_family qwen --prompt_style qwen"
if [[ "$MAKE_LATEST_SYMLINK" == "1" ]]; then
  echo
  echo "Latest symlink:"
  echo "  ${CKPT_PARENT}/latest"
fi
