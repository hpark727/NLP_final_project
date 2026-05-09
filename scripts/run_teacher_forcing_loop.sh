#!/usr/bin/env bash
set -euo pipefail

# True zigzag teacher-forcing loop.
#
# One iteration:
#   1. Teacher mines prefixes.
#      - Iteration 1 seeds from harmful HEx-PHI answers.
#      - Later iterations seed from the previous updated-student mined prefixes.
#   2. Student continues SFT from the previous iteration's training state on the
#      cumulative teacher+student prefix pool.
#   3. Updated student is saved as both training state and sampler weights.
#   4. Updated student mines prefixes; those become the next teacher's seeds and
#      enter the cumulative recovery pool.

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

RUN_NAME="${RUN_NAME:-teacher_forcing_zigzag}"
OUT_DIR="${OUT_DIR:-logs/${RUN_NAME}}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TEACHER_PROMPT_STYLE="${TEACHER_PROMPT_STYLE:-qwen}"
STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.2-3B}"
STUDENT_PROMPT_STYLE="${STUDENT_PROMPT_STYLE:-llama3}"

ITERATIONS="${ITERATIONS:-3}"
MAX_EXAMPLES="${MAX_EXAMPLES:-330}"
SEED_PREFIX_TOKENS="${SEED_PREFIX_TOKENS:-5,10,20}"
SAMPLES_PER_SEED="${SAMPLES_PER_SEED:-3}"
MAX_KEPT_PER_PROMPT="${MAX_KEPT_PER_PROMPT:-2}"
CONTINUATION_TOKENS="${CONTINUATION_TOKENS:-20}"
VERIFICATION_TOKENS="${VERIFICATION_TOKENS:-256}"
MAX_PREFIX_TOKENS="${MAX_PREFIX_TOKENS:-80}"
MINING_CONCURRENCY="${MINING_CONCURRENCY:-16}"

PREFIX_TOKEN_BUDGETS="${PREFIX_TOKEN_BUDGETS:-20,40,full}"
INCLUDE_STATIC_PREFIXES="${INCLUDE_STATIC_PREFIXES:-1}"

SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-1}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-8}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-4}"
SFT_LORA_RANK="${SFT_LORA_RANK:-16}"
SFT_ALPHA="${SFT_ALPHA:-0.2}"
SFT_AUGMENT_PROBABILITY="${SFT_AUGMENT_PROBABILITY:-0.5}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-2048}"
SFT_LOG_EVERY="${SFT_LOG_EVERY:-1}"
LOAD_OPTIMIZER_STATE="${LOAD_OPTIMIZER_STATE:-1}"

SAVE_LOCAL_ADAPTERS="${SAVE_LOCAL_ADAPTERS:-0}"
LOCAL_ADAPTER_DIR="${LOCAL_ADAPTER_DIR:-ckpts/${RUN_NAME}}"
LOCAL_ADAPTER_FORCE_DOWNLOAD="${LOCAL_ADAPTER_FORCE_DOWNLOAD:-0}"
STUDENT_BASE_MODEL_PATH="${STUDENT_BASE_MODEL_PATH:-}"

RUN_ASR_EVAL="${RUN_ASR_EVAL:-1}"
ASR_EVAL_KS="${ASR_EVAL_KS:-5 10 20 40}"
ASR_EVAL_BENCH="${ASR_EVAL_BENCH:-hex-phi_with_harmful_prefix}"
ASR_EVAL_TEMPLATE="${ASR_EVAL_TEMPLATE:-plain}"
ASR_EVAL_MAX_NEW_TOKENS="${ASR_EVAL_MAX_NEW_TOKENS:-512}"
ASR_EVAL_CONCURRENCY="${ASR_EVAL_CONCURRENCY:-16}"
ASR_EVAL_EVALUATOR="${ASR_EVAL_EVALUATOR:-key_word}"
ASR_EVAL_OUTPUT_DISTRIBUTION_STEPS="${ASR_EVAL_OUTPUT_DISTRIBUTION_STEPS:-0}"

mkdir -p "$OUT_DIR"

prefix_paths=()
student_sampler_path=""
student_state_path=""
previous_student_mine_path=""

resolve_student_base_model_path() {
  if [[ -n "$STUDENT_BASE_MODEL_PATH" ]]; then
    echo "$STUDENT_BASE_MODEL_PATH"
    return
  fi

  case "$STUDENT_MODEL" in
    meta-llama/Llama-3.2-3B)
      if [[ -f ckpts/meta-llama/Llama-3.2-3B/config.json ]]; then
        echo "ckpts/meta-llama/Llama-3.2-3B"
      elif [[ -f ckpts/Llama-3.2-3B/config.json ]]; then
        echo "ckpts/Llama-3.2-3B"
      else
        echo "$STUDENT_MODEL"
      fi
      ;;
    Qwen/Qwen3-4B-Instruct-2507)
      if [[ -f ckpts/Qwen3-4B-Instruct-2507/config.json ]]; then
        echo "ckpts/Qwen3-4B-Instruct-2507"
      elif [[ -f ckpts/Qwen/Qwen3-4B-Instruct-2507/config.json ]]; then
        echo "ckpts/Qwen/Qwen3-4B-Instruct-2507"
      else
        echo "$STUDENT_MODEL"
      fi
      ;;
    *)
      echo "$STUDENT_MODEL"
      ;;
  esac
}

patch_adapter_base_model_path() {
  local adapter_dir="$1"
  local base_model_path="$2"

  "$PYTHON_BIN" - "$adapter_dir" "$base_model_path" <<'PY'
import json
import sys
from pathlib import Path

adapter_dir = Path(sys.argv[1])
base_model_path = sys.argv[2]
config_path = adapter_dir / "adapter_config.json"
if not config_path.exists():
    raise SystemExit(f"Missing adapter_config.json in {adapter_dir}")
config = json.loads(config_path.read_text())
config["base_model_name_or_path"] = base_model_path
config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
(adapter_dir / "base_model_path.txt").write_text(base_model_path + "\n")
(adapter_dir / "checkpoint_complete").touch()
PY
}

download_student_adapter() {
  local iter="$1"
  local sampler_path="$2"
  local iter_dir="$3"

  mkdir -p "$LOCAL_ADAPTER_DIR"

  local before_list
  local after_list
  before_list="$(mktemp)"
  after_list="$(mktemp)"

  find "$LOCAL_ADAPTER_DIR" -maxdepth 1 -mindepth 1 -type d -print | sort > "$before_list"

  echo "=== Iteration ${iter}/${ITERATIONS}: download local student adapter ==="
  download_args=(checkpoint download "$sampler_path" --output "$LOCAL_ADAPTER_DIR")
  if [[ "$LOCAL_ADAPTER_FORCE_DOWNLOAD" == "1" ]]; then
    download_args+=(--force)
  fi
  "$TINKER_BIN" "${download_args[@]}"

  find "$LOCAL_ADAPTER_DIR" -maxdepth 1 -mindepth 1 -type d -print | sort > "$after_list"

  local adapter_dir
  adapter_dir="$(comm -13 "$before_list" "$after_list" | while read -r path; do
    if [[ -f "${path}/adapter_config.json" ]]; then
      echo "$path"
      break
    fi
  done)"
  rm -f "$before_list" "$after_list"

  if [[ -z "$adapter_dir" ]]; then
    adapter_dir="$(find "$LOCAL_ADAPTER_DIR" -maxdepth 2 -name adapter_config.json -print \
      | xargs -n1 dirname \
      | sort \
      | tail -1)"
  fi

  if [[ -z "$adapter_dir" || ! -f "${adapter_dir}/adapter_config.json" ]]; then
    echo "Could not find downloaded adapter for iteration ${iter} under ${LOCAL_ADAPTER_DIR}." >&2
    exit 1
  fi

  local base_model_path
  base_model_path="$(resolve_student_base_model_path)"
  patch_adapter_base_model_path "$adapter_dir" "$base_model_path"
  printf "%s\n" "$sampler_path" > "${adapter_dir}/sampler_path.txt"
  printf "%s\n" "$adapter_dir" > "${iter_dir}/student_local_adapter_path.txt"
  ln -sfn "$(basename "$adapter_dir")" "${LOCAL_ADAPTER_DIR}/iter_${iter}"
  ln -sfn "$(basename "$adapter_dir")" "${LOCAL_ADAPTER_DIR}/latest"

  echo "student_local_adapter_path=${adapter_dir}"
}

run_student_asr_eval() {
  local iter="$1"
  local sampler_path="$2"
  local iter_dir="$3"
  local eval_dir="${iter_dir}/asr_eval"

  mkdir -p "$eval_dir"
  echo "=== Iteration ${iter}/${ITERATIONS}: ASR eval k=${ASR_EVAL_KS} ==="

  for k in $ASR_EVAL_KS; do
    "$PYTHON_BIN" eval_safety_tinker.py \
      --model_path "$sampler_path" \
      --prompt_style "$STUDENT_PROMPT_STYLE" \
      --bench "$ASR_EVAL_BENCH" \
      --eval_template "$ASR_EVAL_TEMPLATE" \
      --num_prefix_tokens "$k" \
      --max_new_tokens "$ASR_EVAL_MAX_NEW_TOKENS" \
      --output_distribution_steps "$ASR_EVAL_OUTPUT_DISTRIBUTION_STEPS" \
      --evaluator "$ASR_EVAL_EVALUATOR" \
      --concurrency "$ASR_EVAL_CONCURRENCY" \
      --save_path "${eval_dir}/student_iter_${iter}_${ASR_EVAL_BENCH}_k${k}.json"
  done

  "$PYTHON_BIN" - "$iter" "$eval_dir" "${eval_dir}/student_iter_${iter}_asr_summary.csv" "${OUT_DIR}/asr_summary_all_iterations.csv" <<'PY'
import csv
import json
import sys
from pathlib import Path

iteration = int(sys.argv[1])
eval_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
all_summary_path = Path(sys.argv[4])
rows = []
for path in sorted(eval_dir.glob("student_iter_*_k*.json")):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        metrics = data.get("metrics", data)
        config = data.get("config", {})
    else:
        metrics = {}
        config = {}
    rows.append({
        "iteration": iteration,
        "k": config.get("num_prefix_tokens"),
        "bench": config.get("bench"),
        "file": path.name,
        "asr": metrics.get("asr"),
        "num_tot": metrics.get("num_tot"),
        "num_success": metrics.get("num_success"),
    })

fieldnames = ["iteration", "k", "bench", "asr", "num_success", "num_tot", "file"]
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

all_rows = []
for path in sorted(all_summary_path.parent.glob("iter_*/asr_eval/student_iter_*_asr_summary.csv")):
    with path.open() as handle:
        all_rows.extend(csv.DictReader(handle))
with all_summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
PY
}

parse_checkpoint_paths() {
  local train_log_path="$1"
  local output_json_path="$2"

  "$PYTHON_BIN" - "$train_log_path" "$output_json_path" <<'PY'
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

state_path = checkpoint_paths.get("state_path") or checkpoint_paths.get("state")
sampler_path = checkpoint_paths.get("sampler_path") or checkpoint_paths.get("sampler_weights") or checkpoint_paths.get("sampler")

if state_path:
    (save_path.parent / "student_state_path.txt").write_text(str(state_path) + "\n")
    print(f"student_state_path={state_path}")
if sampler_path:
    (save_path.parent / "student_sampler_path.txt").write_text(str(sampler_path) + "\n")
    print(f"student_sampler_path={sampler_path}")
PY
}

asr_eval_complete() {
  local iter="$1"
  local iter_dir="$2"
  local eval_dir="${iter_dir}/asr_eval"

  for k in $ASR_EVAL_KS; do
    if [[ ! -s "${eval_dir}/student_iter_${iter}_${ASR_EVAL_BENCH}_k${k}.json" ]]; then
      return 1
    fi
  done
  return 0
}

mine_prefixes() {
  local role="$1"
  local iter="$2"
  local model_arg_name="$3"
  local model_value="$4"
  local prompt_style="$5"
  local save_path="$6"
  local metadata_path="$7"
  local all_candidates_path="$8"
  local seed_prefix_path="${9:-}"

  local args=(
    mine_adaptive_prefixes_tinker.py
    "$model_arg_name" "$model_value"
    --prompt_style "$prompt_style"
    --eval_template null
    --max_examples "$MAX_EXAMPLES"
    --seed_prefix_tokens "$SEED_PREFIX_TOKENS"
    --samples_per_seed "$SAMPLES_PER_SEED"
    --max_kept_per_prompt "$MAX_KEPT_PER_PROMPT"
    --continuation_tokens "$CONTINUATION_TOKENS"
    --verification_tokens "$VERIFICATION_TOKENS"
    --max_prefix_tokens "$MAX_PREFIX_TOKENS"
    --random_seed "$iter"
    --concurrency "$MINING_CONCURRENCY"
    --save_path "$save_path"
    --metadata_path "$metadata_path"
    --save_all_candidates_path "$all_candidates_path"
  )

  if [[ -n "$seed_prefix_path" ]]; then
    args+=(--seed_prefix_path "$seed_prefix_path" --seed_prefix_field prefix)
  fi

  echo "=== Iteration ${iter}/${ITERATIONS}: mine ${role} prefixes ==="
  "$PYTHON_BIN" "${args[@]}"
}

for iter in $(seq 1 "$ITERATIONS"); do
  iter_dir="${OUT_DIR}/iter_${iter}"
  mkdir -p "$iter_dir"

  teacher_mine_path="${iter_dir}/mined_teacher_iter_${iter}.jsonl"
  teacher_all_path="${iter_dir}/all_candidates_teacher_iter_${iter}.jsonl"
  teacher_meta_path="${iter_dir}/mined_teacher_iter_${iter}.metadata.json"
  student_mine_path="${iter_dir}/mined_student_iter_${iter}.jsonl"
  student_all_path="${iter_dir}/all_candidates_student_iter_${iter}.jsonl"
  student_meta_path="${iter_dir}/mined_student_iter_${iter}.metadata.json"
  data_dir="${iter_dir}/data_cumulative_zigzag"
  train_log_path="${iter_dir}/student_sft.log"
  checkpoint_dir="${iter_dir}/student_sft_checkpoint"
  mkdir -p "$checkpoint_dir"

  if [[ -s "$teacher_mine_path" ]]; then
    echo "=== Iteration ${iter}/${ITERATIONS}: reuse existing teacher prefixes ==="
    echo "teacher_mine_path=${teacher_mine_path}"
  else
    mine_prefixes \
      "teacher" \
      "$iter" \
      --model_name \
      "$TEACHER_MODEL" \
      "$TEACHER_PROMPT_STYLE" \
      "$teacher_mine_path" \
      "$teacher_meta_path" \
      "$teacher_all_path" \
      "$previous_student_mine_path"
  fi

  prefix_paths+=("$teacher_mine_path")

  if [[ -s "${data_dir}/train.jsonl" ]]; then
    echo "=== Iteration ${iter}/${ITERATIONS}: reuse existing cumulative recovery data ==="
    echo "train_data=${data_dir}/train.jsonl"
  else
    echo "=== Iteration ${iter}/${ITERATIONS}: prepare cumulative recovery data ==="
    prepare_args=(
      prepare_adaptive_recovery_data.py
      --prefix_paths "${prefix_paths[@]}"
      --prefix_token_budgets "$PREFIX_TOKEN_BUDGETS"
      --save_dir "$data_dir"
      --seed "$iter"
    )
    if [[ "$INCLUDE_STATIC_PREFIXES" == "1" ]]; then
      prepare_args+=(--include_static_prefixes)
    fi
    "$PYTHON_BIN" "${prepare_args[@]}"
  fi

  if [[ -s "${iter_dir}/student_sampler_path.txt" && -s "${iter_dir}/student_state_path.txt" ]]; then
    echo "=== Iteration ${iter}/${ITERATIONS}: reuse existing student checkpoint paths ==="
  else
    echo "=== Iteration ${iter}/${ITERATIONS}: continue-train student ==="
    train_args=(
      finetune_tinker.py
      --model_name "$STUDENT_MODEL"
      --safety_data_path "${data_dir}/train.jsonl"
      --augmentation_prefix_field prefix
      --augmentation_prefix_tokens 0
      --augment_probability "$SFT_AUGMENT_PROBABILITY"
      --num_epochs "$SFT_NUM_EPOCHS"
      --batch_size "$SFT_BATCH_SIZE"
      --learning_rate "$SFT_LEARNING_RATE"
      --lora_rank "$SFT_LORA_RANK"
      --alpha "$SFT_ALPHA"
      --max_length "$SFT_MAX_LENGTH"
      --log_every "$SFT_LOG_EVERY"
      --log_path "$checkpoint_dir"
    )
    if [[ -n "$student_state_path" ]]; then
      train_args+=(--load_state_path "$student_state_path")
      if [[ "$LOAD_OPTIMIZER_STATE" == "1" ]]; then
        train_args+=(--load_optimizer_state)
      fi
    fi

    PYTHONUNBUFFERED=1 "$PYTHON_BIN" "${train_args[@]}" 2>&1 | tee "$train_log_path"
    parse_checkpoint_paths "$train_log_path" "${iter_dir}/student_checkpoint_paths.json"
  fi

  if [[ -f "${iter_dir}/student_sampler_path.txt" ]]; then
    student_sampler_path="$(<"${iter_dir}/student_sampler_path.txt")"
  else
    echo "Could not find student sampler path after iteration ${iter}." >&2
    exit 1
  fi
  if [[ -f "${iter_dir}/student_state_path.txt" ]]; then
    student_state_path="$(<"${iter_dir}/student_state_path.txt")"
  else
    echo "Could not find student state path after iteration ${iter}." >&2
    exit 1
  fi

  if [[ "$SAVE_LOCAL_ADAPTERS" == "1" ]]; then
    if [[ -s "${iter_dir}/student_local_adapter_path.txt" ]]; then
      echo "=== Iteration ${iter}/${ITERATIONS}: reuse existing local student adapter ==="
      echo "student_local_adapter_path=$(<"${iter_dir}/student_local_adapter_path.txt")"
    else
      download_student_adapter "$iter" "$student_sampler_path" "$iter_dir"
    fi
  fi

  if [[ "$RUN_ASR_EVAL" == "1" ]]; then
    if asr_eval_complete "$iter" "$iter_dir"; then
      echo "=== Iteration ${iter}/${ITERATIONS}: reuse existing ASR eval ==="
    else
      run_student_asr_eval "$iter" "$student_sampler_path" "$iter_dir"
    fi
  fi

  if [[ -s "$student_mine_path" ]]; then
    echo "=== Iteration ${iter}/${ITERATIONS}: reuse existing updated-student prefixes ==="
    echo "student_mine_path=${student_mine_path}"
  else
    mine_prefixes \
      "updated-student" \
      "$iter" \
      --model_path \
      "$student_sampler_path" \
      "$STUDENT_PROMPT_STYLE" \
      "$student_mine_path" \
      "$student_meta_path" \
      "$student_all_path" \
      "$teacher_mine_path"
  fi

  prefix_paths+=("$student_mine_path")
  previous_student_mine_path="$student_mine_path"

  echo "=== Iteration ${iter}/${ITERATIONS}: done ==="
  echo "student_sampler_path=${student_sampler_path}"
  echo "student_state_path=${student_state_path}"
done

echo "All zigzag iterations complete."
echo "Outputs written under: ${OUT_DIR}"
