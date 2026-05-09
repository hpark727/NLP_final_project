"""
Logit trajectory across response steps 0-15.

Captures refusal/compliance log-masses and refusal margin at each greedy
generation step without storing hidden states — much faster than re-running
refusal_trajectory_multidim.py with --response_steps 0,...,15.

Output: logs/adaptive_prefixes/logit_traj/logit_trajectory.csv
  columns: condition, k, example_idx, response_step,
           refusal_logmass, compliance_logmass, refusal_margin
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from finetuning_buckets.inference import chat
from finetuning_buckets.inference.safety_eval.evaluator import common_eval_template
from finetuning_buckets.models import get_model

# ── match the existing traj_multidim run exactly ──────────────────────────────
_META = json.load(open("logs/adaptive_prefixes/traj_multidim/metadata.json"))
MODEL_PATH          = _META["model_name_or_path"]
ADAPTIVE_PREFIX_PATH = _META["args"]["adaptive_prefix_path"]
HARMFUL_PREFIX_PATH  = _META["args"]["harmful_prefix_path"]
REFUSAL_PREFIX_PATH  = _META["args"]["refusal_prefix_path"]
REFUSAL_FALLBACK     = _META["args"]["refusal_fallback"]
KS                   = [5, 40]
MAX_EXAMPLES         = _META["args"]["max_examples"]
SEED                 = _META["args"]["seed"]
BATCH_SIZE           = _META["args"]["batch_size"]
MAX_LENGTH           = _META["args"]["max_length"]
PROMPT_STYLE         = _META["args"]["prompt_style"]
MODEL_FAMILY         = _META["args"]["model_family"]
EVAL_TEMPLATE_KEY    = _META["args"]["eval_template"]
REFUSAL_TOKEN_IDS    = _META["refusal_token_ids"]
COMPLIANCE_TOKEN_IDS = _META["compliance_token_ids"]

MAX_STEPS = 40   # measure steps 0..40 inclusive

SAVE_DIR = Path("logs/adaptive_prefixes/logit_traj")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = SAVE_DIR / "logit_trajectory.csv"


# ── helpers (copied from refusal_trajectory_multidim.py) ─────────────────────

def encode_no_special(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def first_k_tokens(tokenizer, text: str, k: int) -> str:
    return tokenizer.decode(encode_no_special(tokenizer, text)[:k], skip_special_tokens=False)


def load_adaptive_records(path: str, max_examples: int, seed: int) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    random.Random(seed).shuffle(records)
    return records[:max_examples] if max_examples > 0 else records


def load_pair_map(path: str) -> dict[str, str]:
    pairs = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, list) and len(item) >= 2:
                pairs[item[0]["content"]] = item[1]["content"]
            elif isinstance(item, dict) and "instruction" in item and "harmful" in item:
                pairs[item["instruction"]] = item["harmful"]
    return pairs


def build_messages(system_prompt, input_template, instruction, assistant_prefix):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    user_content = input_template % instruction if input_template else instruction
    msgs.append({"role": "user", "content": user_content})
    msgs.append({"role": "assistant", "content": assistant_prefix})
    return msgs


def render_messages(generator, messages):
    conversation = generator.validate_conversation(messages)
    return generator.string_formatter({"messages": conversation})["text"]


# ── core: batched greedy generation with per-step logit capture ───────────────

def generate_with_logit_capture(
    model,
    tokenizer,
    prompts: list[str],
    max_steps: int,
    batch_size: int,
    max_length: int,
    refusal_token_ids: list[int],
    compliance_token_ids: list[int],
    desc: str,
) -> list[dict]:
    device = next(model.parameters()).device
    r_ids = torch.tensor(refusal_token_ids, device=device)
    c_ids = torch.tensor(compliance_token_ids, device=device)
    max_prompt_length = max(1, max_length - max_steps)

    rows = []
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[batch_start : batch_start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)

        # Strip padding; tokenizer has padding_side="left" so real tokens are
        # always at the right end — logits[:, -1, :] is always the last real token.
        attn = enc["attention_mask"]
        current_ids: list[list[int]] = [
            enc["input_ids"][i, attn[i].bool()].tolist()
            for i in range(len(batch))
        ]

        for step in range(max_steps + 1):
            step_enc = tokenizer.pad(
                {"input_ids": current_ids},
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                out = model(
                    input_ids=step_enc["input_ids"],
                    attention_mask=step_enc["attention_mask"],
                    use_cache=False,
                )

            logits = out.logits[:, -1, :].float()   # [B, vocab]
            r_mass = torch.logsumexp(logits[:, r_ids], dim=-1).cpu().numpy()
            c_mass = torch.logsumexp(logits[:, c_ids], dim=-1).cpu().numpy()

            for bi in range(len(batch)):
                rows.append({
                    "example_idx":       batch_start + bi,
                    "response_step":     step,
                    "refusal_logmass":   float(r_mass[bi]),
                    "compliance_logmass": float(c_mass[bi]),
                    "refusal_margin":    float(r_mass[bi] - c_mass[bi]),
                })

            if step < max_steps:
                next_tokens = logits.argmax(dim=-1).tolist()
                for bi, tok in enumerate(next_tokens):
                    current_ids[bi].append(tok)

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    model_kwargs = dict(
        torch_dtype=torch.float16,
        use_cache=False,
    )
    model, tokenizer = get_model.get_model(
        MODEL_PATH,
        model_kwargs,
        model_family=MODEL_FAMILY,
        padding_side="left",
    )
    model.eval()
    if torch.backends.mps.is_available():
        model.to(torch.device("mps"))

    eval_tmpl   = common_eval_template[EVAL_TEMPLATE_KEY]
    system_prompt   = eval_tmpl["system_prompt"]
    input_template  = eval_tmpl["input_template"]
    generator = chat.Chat(
        model=model, prompt_style=PROMPT_STYLE,
        tokenizer=tokenizer, init_system_prompt=system_prompt,
    )

    adaptive_records = load_adaptive_records(ADAPTIVE_PREFIX_PATH, MAX_EXAMPLES, SEED)
    harmful_map  = load_pair_map(HARMFUL_PREFIX_PATH)
    refusal_map  = load_pair_map(REFUSAL_PREFIX_PATH) if Path(REFUSAL_PREFIX_PATH).exists() else {}

    matched = [r for r in adaptive_records if r["instruction"] in harmful_map]
    random.Random(SEED).shuffle(matched)
    print(f"{len(matched)} matched examples, max_steps={MAX_STEPS}, ks={KS}")

    all_rows: list[dict] = []
    conditions = ["clean", "refusal", "static", "adaptive"]

    for k in KS:
        print(f"\n{'='*60}\nk = {k}\n{'='*60}")
        rendered: dict[str, list[str]] = {c: [] for c in conditions}

        for record in matched:
            instr = record["instruction"]
            prefixes = {
                "clean":    "",
                "refusal":  first_k_tokens(tokenizer, refusal_map.get(instr, REFUSAL_FALLBACK), k),
                "static":   first_k_tokens(tokenizer, harmful_map[instr], k),
                "adaptive": first_k_tokens(tokenizer, record["prefix"], k),
            }
            for cond, prefix in prefixes.items():
                msgs = build_messages(system_prompt, input_template, instr, prefix)
                rendered[cond].append(render_messages(generator, msgs))

        for cond in conditions:
            rows = generate_with_logit_capture(
                model, tokenizer,
                rendered[cond],
                max_steps=MAX_STEPS,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH,
                refusal_token_ids=REFUSAL_TOKEN_IDS,
                compliance_token_ids=COMPLIANCE_TOKEN_IDS,
                desc=f"k={k} {cond}",
            )
            for row in rows:
                row["k"] = k
                row["condition"] = cond
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df):,} rows → {OUT_CSV}")
    print(df.groupby(["condition", "k"])["refusal_margin"].describe().round(3))


if __name__ == "__main__":
    main()
