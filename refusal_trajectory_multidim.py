from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from finetuning_buckets.inference import chat
from finetuning_buckets.inference.safety_eval.evaluator import common_eval_template
from finetuning_buckets.models import get_model


DEFAULT_HARMFUL_PATH = "finetuning_buckets/datasets/data/safety_bench/Harmful-HEx-PHI.jsonl"
DEFAULT_REFUSAL_PATH = "finetuning_buckets/datasets/data/safety_bench/llama2_HEx-PHI_refusal_examples.jsonl"
DEFAULT_BENIGN_PATH = "finetuning_buckets/datasets/data/tasks/data_augmentation/llama2_alpaca_anchor.json"


@dataclass
class ScriptArguments:
    model_family: str = field(default="llama3")
    prompt_style: str = field(default="llama3")
    eval_template: str = field(default="null")
    adaptive_prefix_path: str = field(
        default="logs/adaptive_prefixes/mined_augmented_tinker.jsonl",
        metadata={"help": "JSONL with `instruction` and `prefix` fields."},
    )
    harmful_prefix_path: str = field(default=DEFAULT_HARMFUL_PATH)
    refusal_prefix_path: str = field(default=DEFAULT_REFUSAL_PATH)
    benign_path: str = field(default=DEFAULT_BENIGN_PATH)
    refusal_fallback: str = field(default="I cannot fulfill this request.")
    ks: str = field(default="5,10,20,40")
    max_examples: int = field(default=128, metadata={"help": "Max adaptive examples to analyze; <=0 uses all."})
    max_probe_samples_per_class: int = field(default=128)
    probe_train_fraction: float = field(default=0.7)
    probe_steps: int = field(default=400)
    probe_lr: float = field(default=0.05)
    probe_l2_weight: float = field(default=0.01)
    batch_size: int = field(default=2)
    max_length: int = field(default=2048)
    response_steps: str = field(
        default="0,1,2,4,8,16",
        metadata={
            "help": (
                "Comma-separated response steps after the prefill to probe. "
                "0 probes immediately after the prefill, 1 after one generated token."
            )
        },
    )
    layers: str = field(
        default="all",
        metadata={
            "help": (
                "Comma-separated hidden-state layers to keep, or `all`/`final`. "
                "Negative indices count from the end, e.g. -1 is the final layer."
            )
        },
    )
    seed: int = field(default=0)
    save_dir: str = field(default="logs/adaptive_prefixes/traj_anal_per_token")
    train_harmfulness_probe: bool = field(default=True)
    compute_kl: bool = field(
        default=True,
        metadata={"help": "Compute full-vocabulary KL to the refusal next-token distribution."},
    )
    kl_reference_condition: str = field(
        default="refusal",
        metadata={"help": "Reference condition for KL plots/columns: clean, refusal, static, or adaptive."},
    )
    baseline_condition: str = field(
        default="clean",
        metadata={"help": "Condition used as the paired baseline for hidden-state deltas/subspace projections."},
    )
    refusal_subspace_rank: int = field(
        default=8,
        metadata={"help": "Rank r for the PCA/SVD refusal subspace built from baseline-to-refusal deltas."},
    )
    condition_subspace_rank: int = field(
        default=8,
        metadata={"help": "Rank r for static/adaptive/refusal subspaces used in principal-angle analysis."},
    )
    center_subspace_deltas: bool = field(
        default=False,
        metadata={"help": "If true, mean-center deltas before SVD. Default false keeps the mean refusal direction in the subspace."},
    )
    compute_principal_angles: bool = field(
        default=True,
        metadata={"help": "Compute principal angles between refusal/static/adaptive hidden-state delta subspaces."},
    )
    principal_angle_conditions: str = field(
        default="refusal,static,adaptive",
        metadata={"help": "Comma-separated conditions whose baseline-relative subspaces are compared via principal angles."},
    )
    use_cache: bool = field(default=False)
    device: str = field(default="auto", metadata={"help": "Device to use: auto, cuda, mps, or cpu."})
    prefer_mps_float16: bool = field(
        default=True,
        metadata={"help": "Use float16 by default on Apple MPS when --torch_dtype is unset/auto."},
    )
    refusal_token_strings: str = field(
        default="I, sorry, Sorry, cannot, can't, unable, apologize, I cannot",
        metadata={"help": "Comma-separated strings whose first token contributes to refusal mass."},
    )
    compliance_token_strings: str = field(
        default="Step, First, 1, Sure, Here, To, The",
        metadata={"help": "Comma-separated strings whose first token contributes to compliance mass."},
    )


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA, but torch.cuda.is_available() is false")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS, but torch.backends.mps.is_available() is false")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device {device}; expected auto, cuda, mps, or cpu")
    return torch.device(requested)


def parse_layer_selector(raw: str) -> str | list[int]:
    value = raw.strip().lower()
    if value in {"all", "*"}:
        return "all"
    if value in {"final", "last"}:
        return [-1]
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not indices:
        raise ValueError("Expected --layers to be all, final, or a comma-separated integer list")
    return indices


def resolve_layer_indices(selector: str | list[int], num_layers: int) -> list[int]:
    if selector == "all":
        return list(range(num_layers))
    resolved = []
    for layer in selector:
        index = layer if layer >= 0 else num_layers + layer
        if index < 0 or index >= num_layers:
            raise ValueError(f"Layer index {layer} is out of range for {num_layers} hidden states")
        resolved.append(index)
    return sorted(set(resolved))


def encode_no_special(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def first_k_tokens(tokenizer, text: str, k: int) -> str:
    return tokenizer.decode(encode_no_special(tokenizer, text)[:k], skip_special_tokens=False)


def first_token_ids(tokenizer, raw: str) -> list[int]:
    ids = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        token_ids = encode_no_special(tokenizer, item)
        if token_ids:
            ids.append(int(token_ids[0]))
    return sorted(set(ids))


def load_adaptive_records(path: str, max_examples: int, seed: int) -> list[dict[str, Any]]:
    records = []
    with open(path) as handle:
        for line_idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if "instruction" not in record or "prefix" not in record:
                raise ValueError(f"Line {line_idx + 1} in {path} needs `instruction` and `prefix`")
            records.append(record)
    rng = random.Random(seed)
    rng.shuffle(records)
    if max_examples > 0:
        records = records[:max_examples]
    return records


def load_pair_map(path: str) -> dict[str, str]:
    pairs = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, list) and len(item) >= 2:
                pairs[item[0]["content"]] = item[1]["content"]
            elif isinstance(item, dict) and "instruction" in item and "harmful" in item:
                pairs[item["instruction"]] = item["harmful"]
    return pairs


def load_benign_prompts(path: str, max_samples: int, seed: int) -> list[str]:
    with open(path) as handle:
        data = json.load(handle)
    prompts = []
    for item in data:
        instruction = item["instruction"]
        if item.get("input", "").strip():
            instruction = f"{instruction}\n\n{item['input']}"
        prompts.append(instruction)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:max_samples] if max_samples > 0 else prompts


def load_harmful_prompts(path: str, max_samples: int, seed: int) -> list[str]:
    prompts = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, list):
                prompts.append(item[0]["content"])
            else:
                prompts.append(item["instruction"])
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:max_samples] if max_samples > 0 else prompts


def build_messages(
    system_prompt: str | None,
    input_template: str | None,
    instruction: str,
    assistant_prefix: str,
) -> list[dict[str, Any]]:
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    user_content = input_template % instruction if input_template is not None else instruction
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_prefix})
    return messages


def render_messages(generator: chat.Chat, messages: list[dict[str, Any]]) -> str:
    conversation = generator.validate_conversation(messages)
    return generator.string_formatter({"messages": conversation})["text"]


def tokenization_kwargs(max_length: int, offsets: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"padding": False}
    if offsets:
        kwargs["return_offsets_mapping"] = True
    if max_length > 0:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_length
    return kwargs


def find_last_user_token(prompt: str, instruction: str, tokenizer, max_length: int) -> int:
    encoded = tokenizer(prompt, **tokenization_kwargs(max_length, offsets=True))
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    start = prompt.rfind(instruction)
    if start < 0:
        return max(0, len(encoded["input_ids"]) - 1)
    end = start + len(instruction)
    indices = [
        idx
        for idx, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < end and token_end > start
    ]
    return indices[-1] if indices else max(0, len(encoded["input_ids"]) - 1)


def selected_positions_from_unpadded(
    attention_mask: torch.Tensor,
    token_positions: torch.Tensor,
    padding_side: str,
) -> torch.Tensor:
    token_positions = token_positions.to(attention_mask.device)
    if padding_side == "left":
        sequence_lengths = attention_mask.long().sum(dim=1)
        return token_positions + (attention_mask.shape[1] - sequence_lengths)
    return token_positions


def normalize_position_lists(token_positions: list[int] | list[list[int]]) -> list[list[int]]:
    if not token_positions:
        return []
    first = token_positions[0]
    if isinstance(first, int):
        return [[int(position)] for position in token_positions]  # type: ignore[arg-type]
    position_lists = [[int(position) for position in positions] for positions in token_positions]  # type: ignore[union-attr]
    widths = {len(positions) for positions in position_lists}
    if len(widths) != 1:
        raise ValueError("All examples in a forward pass must have the same number of probe positions")
    return position_lists


def prepare_model_inputs(tokenizer, batch_inputs: list[str] | list[list[int]], max_length: int, device: torch.device):
    if not batch_inputs:
        raise ValueError("Cannot prepare an empty batch")
    if isinstance(batch_inputs[0], str):
        return tokenizer(
            batch_inputs,
            padding=True,
            return_tensors="pt",
            truncation=max_length > 0,
            max_length=max_length if max_length > 0 else None,
        ).to(device)
    return tokenizer.pad(
        {"input_ids": batch_inputs},
        padding=True,
        return_tensors="pt",
    ).to(device)


def selected_positions_from_unpadded_matrix(
    attention_mask: torch.Tensor,
    token_positions: torch.Tensor,
    padding_side: str,
) -> torch.Tensor:
    token_positions = token_positions.to(attention_mask.device)
    if padding_side == "left":
        sequence_lengths = attention_mask.long().sum(dim=1)
        padding_offsets = attention_mask.shape[1] - sequence_lengths
        return token_positions + padding_offsets[:, None]
    return token_positions


def forward_selected(
    model,
    tokenizer,
    prompts: list[str] | list[list[int]],
    token_positions: list[int] | list[list[int]],
    batch_size: int,
    max_length: int,
    layer_selector: str | list[int],
    use_cache: bool,
    desc: str,
    *,
    collect_logprobs: bool,
    refusal_token_ids: list[int] | None = None,
    compliance_token_ids: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[int]]:
    model_for_forward = getattr(model, "module", model)
    device = next(model_for_forward.parameters()).device
    position_lists = normalize_position_lists(token_positions)
    feature_chunks = []
    logprob_chunks = []
    refusal_logmass_chunks = []
    compliance_logmass_chunks = []
    selected_layer_ids: list[int] | None = None

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start : start + batch_size]
        batch_positions = torch.tensor(position_lists[start : start + batch_size], dtype=torch.long)
        model_inputs = prepare_model_inputs(tokenizer, batch_prompts, max_length, device)
        with torch.inference_mode():
            outputs = model_for_forward(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                output_hidden_states=True,
                use_cache=use_cache,
            )
            selected_positions = selected_positions_from_unpadded_matrix(
                model_inputs["attention_mask"],
                batch_positions,
                tokenizer.padding_side,
            )
            batch_indices = torch.arange(model_inputs["attention_mask"].shape[0], device=device)
            gather_batch_indices = batch_indices[:, None]
            if selected_layer_ids is None:
                selected_layer_ids = resolve_layer_indices(layer_selector, len(outputs.hidden_states))
            features = torch.stack(
                [
                    hidden_state[gather_batch_indices, selected_positions]
                    for layer_idx, hidden_state in enumerate(outputs.hidden_states)
                    if layer_idx in selected_layer_ids
                ],
                dim=2,
            )
            feature_chunks.append(features.float().cpu().numpy().astype(np.float32))

            if collect_logprobs:
                logits = outputs.logits[gather_batch_indices, selected_positions].float()
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                logprob_chunks.append(logprobs.cpu().numpy().astype(np.float32))
            elif refusal_token_ids or compliance_token_ids:
                logits = outputs.logits[gather_batch_indices, selected_positions].float()
                log_denom = torch.logsumexp(logits, dim=-1)
                if refusal_token_ids:
                    token_logits = logits[:, refusal_token_ids]
                    refusal_logmass = torch.logsumexp(token_logits, dim=-1) - log_denom
                    refusal_logmass_chunks.append(refusal_logmass.cpu().numpy().astype(np.float32))
                if compliance_token_ids:
                    token_logits = logits[:, compliance_token_ids]
                    compliance_logmass = torch.logsumexp(token_logits, dim=-1) - log_denom
                    compliance_logmass_chunks.append(compliance_logmass.cpu().numpy().astype(np.float32))

    features_np = np.concatenate(feature_chunks, axis=0)
    logprobs_np = np.concatenate(logprob_chunks, axis=0) if collect_logprobs else None
    refusal_logmass_np = (
        np.concatenate(refusal_logmass_chunks, axis=0)
        if refusal_logmass_chunks
        else None
    )
    compliance_logmass_np = (
        np.concatenate(compliance_logmass_chunks, axis=0)
        if compliance_logmass_chunks
        else None
    )
    if selected_layer_ids is None:
        selected_layer_ids = []
    return features_np, logprobs_np, refusal_logmass_np, compliance_logmass_np, selected_layer_ids


def generate_greedy_contexts(
    model,
    tokenizer,
    prompts: list[str],
    response_steps: list[int],
    batch_size: int,
    max_length: int,
    desc: str,
) -> tuple[list[list[int]], list[list[int]]]:
    max_step = max(response_steps)
    model_for_forward = getattr(model, "module", model)
    device = next(model_for_forward.parameters()).device
    generated_contexts: list[list[int]] = []
    selected_positions: list[list[int]] = []
    max_prompt_length = max_length - max_step if max_length > max_step else max_length

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start : start + batch_size]
        model_inputs = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            truncation=max_prompt_length > 0,
            max_length=max_prompt_length if max_prompt_length > 0 else None,
        ).to(device)
        attention_mask = model_inputs["attention_mask"]
        current_ids = [
            model_inputs["input_ids"][row_idx, attention_mask[row_idx].bool()].tolist()
            for row_idx in range(attention_mask.shape[0])
        ]

        for _ in range(max_step):
            step_inputs = tokenizer.pad(
                {"input_ids": current_ids},
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                outputs = model_for_forward(
                    input_ids=step_inputs["input_ids"],
                    attention_mask=step_inputs["attention_mask"],
                    use_cache=False,
                )
                next_tokens = outputs.logits[:, -1, :].argmax(dim=-1).tolist()
            for row_idx, next_token in enumerate(next_tokens):
                current_ids[row_idx].append(int(next_token))

        generated_contexts.extend(current_ids)
        selected_positions.extend(
            [[max(0, len(input_ids) - max_step - 1 + step) for step in response_steps] for input_ids in current_ids]
        )

    return generated_contexts, selected_positions


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def train_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    args: ScriptArguments,
    seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    x_eval = torch.tensor(eval_x, dtype=torch.float32)

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train_norm = (x_train - mean) / std
    x_eval_norm = (x_eval - mean) / std

    linear = torch.nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=args.probe_lr, weight_decay=0.0)
    for _ in range(args.probe_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = linear(x_train_norm).squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(logits, y_train)
        l2 = linear.weight.square().sum()
        loss = bce + args.probe_l2_weight * l2 / x_train.shape[0]
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        train_scores = torch.sigmoid(linear(x_train_norm).squeeze(-1)).numpy()
        eval_scores = torch.sigmoid(linear(x_eval_norm).squeeze(-1)).numpy()

    predictions = (eval_scores >= 0.5).astype(np.int64)
    metrics = {
        "train_auroc": binary_auc(train_y, train_scores),
        "eval_auroc": binary_auc(eval_y, eval_scores),
        "eval_accuracy": float((predictions == eval_y).mean()),
    }
    state = {
        "weight": linear.weight.detach().numpy().astype(np.float32),
        "bias": linear.bias.detach().numpy().astype(np.float32),
        "mean": mean.numpy().astype(np.float32),
        "std": std.numpy().astype(np.float32),
    }
    return metrics, state


def apply_probe(features: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    x = (features - state["mean"]) / state["std"]
    logits = x @ state["weight"].T + state["bias"]
    return 1.0 / (1.0 + np.exp(-logits.squeeze(-1)))


def stratified_split(labels: np.ndarray, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train_indices = []
    eval_indices = []
    for label in sorted(set(labels.tolist())):
        indices = [idx for idx, value in enumerate(labels.tolist()) if value == label]
        rng.shuffle(indices)
        train_count = max(1, min(len(indices) - 1, int(round(len(indices) * train_fraction))))
        train_indices.extend(indices[:train_count])
        eval_indices.extend(indices[train_count:])
    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)
    return train_indices, eval_indices


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    # Some rows, especially principal-angle rows, may have angle_1...angle_r
    # columns only when the corresponding basis has enough rank. Use the union
    # of keys so later rows do not crash DictWriter.
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(path)




def parse_condition_list(raw: str, allowed: list[str]) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one condition")
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"Unknown condition(s) {unknown}; expected only {allowed}")
    return values


def safe_vector_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(x, axis=axis)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float64)
    mask = denominator > 1e-8
    out[mask] = numerator[mask] / denominator[mask]
    return out


def orthonormal_basis_from_deltas(
    deltas: np.ndarray,
    rank: int,
    *,
    center: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a rank-r right-singular-vector basis for rows of deltas.

    deltas has shape [n_examples, hidden_dim]. With center=False, the SVD is run on
    raw baseline-to-condition deltas, so the dominant mean trajectory is retained.
    """
    if deltas.ndim != 2:
        raise ValueError(f"Expected 2D deltas, got shape {deltas.shape}")
    n_examples, hidden_dim = deltas.shape
    max_rank = min(max(rank, 0), n_examples, hidden_dim)
    if max_rank == 0:
        return np.zeros((hidden_dim, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0

    matrix = deltas.astype(np.float64, copy=True)
    if center:
        matrix -= matrix.mean(axis=0, keepdims=True)
    if not np.isfinite(matrix).all() or np.linalg.norm(matrix) < 1e-12:
        return np.zeros((hidden_dim, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0

    try:
        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        matrix = matrix + 1e-8 * np.random.default_rng(0).standard_normal(matrix.shape)
        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)

    keep = min(max_rank, len(singular_values))
    basis = vt[:keep].T.astype(np.float32)
    kept_singular_values = singular_values[:keep].astype(np.float32)
    total_energy = float(np.square(singular_values).sum())
    kept_energy = float(np.square(singular_values[:keep]).sum())
    explained = kept_energy / total_energy if total_energy > 1e-12 else 0.0
    return basis, kept_singular_values, explained


def project_metrics_for_basis(deltas: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute projection norm, projection fraction, and residual norm for each row."""
    delta_norm = safe_vector_norm(deltas, axis=-1).astype(np.float64)
    if basis.size == 0:
        projection_norm = np.zeros_like(delta_norm)
    else:
        coords = deltas.astype(np.float64) @ basis.astype(np.float64)
        projection_norm = safe_vector_norm(coords, axis=-1).astype(np.float64)
    projection_norm = np.minimum(projection_norm, delta_norm + 1e-6)
    projection_fraction = safe_divide(projection_norm, delta_norm)
    residual_sq = np.maximum(np.square(delta_norm) - np.square(projection_norm), 0.0)
    residual_norm = np.sqrt(residual_sq)
    return projection_norm, projection_fraction, residual_norm


def cosine_to_reference(deltas: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Row-wise cosine between deltas and references. Zero when either vector is near zero."""
    numerator = np.sum(deltas.astype(np.float64) * references.astype(np.float64), axis=-1)
    denominator = safe_vector_norm(deltas, axis=-1) * safe_vector_norm(references, axis=-1)
    return safe_divide(numerator, denominator)


def principal_angles_degrees(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles in degrees between two orthonormal bases."""
    if basis_a.size == 0 or basis_b.size == 0:
        return np.array([], dtype=np.float64)
    singular_values = np.linalg.svd(
        basis_a.astype(np.float64).T @ basis_b.astype(np.float64),
        compute_uv=False,
    )
    singular_values = np.clip(singular_values, -1.0, 1.0)
    return np.degrees(np.arccos(singular_values))

def aggregate_rows(
    per_example_rows: list[dict[str, Any]],
    include_probe: bool,
    include_kl: bool,
    kl_metric_name: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str, int], list[dict[str, Any]]] = {}
    for row in per_example_rows:
        key = (row["k"], row["response_step"], row["condition"], row["layer"])
        groups.setdefault(key, []).append(row)

    summary = []
    metric_names = [
        "safety_projection",  # backwards-compatible alias for refusal_mean_direction_projection
        "delta_norm_from_baseline",
        "refusal_mean_direction_projection",
        "refusal_subspace_projection_norm",
        "refusal_subspace_fraction",
        "refusal_subspace_residual_norm",
        "cosine_to_refusal_delta",
        "cosine_to_refusal_mean_direction",
        "refusal_subspace_rank_used",
        "refusal_subspace_explained_variance",
        "refusal_logmass",
        "compliance_logmass",
        "refusal_margin",
    ]
    if include_probe:
        metric_names.insert(1, "harmfulness_probe_score")
    if include_kl:
        insert_at = 2 if include_probe else 1
        metric_names.insert(insert_at, kl_metric_name)
    for (k, response_step, condition, layer), rows in sorted(groups.items()):
        item: dict[str, Any] = {
            "k": k,
            "response_step": response_step,
            "condition": condition,
            "layer": layer,
            "n": len(rows),
        }
        for metric_name in metric_names:
            values = np.array(
                [row.get(metric_name, "") for row in rows if row.get(metric_name, "") != ""],
                dtype=np.float64,
            )
            if len(values) == 0:
                item[f"{metric_name}_mean"] = ""
                item[f"{metric_name}_std"] = ""
            else:
                item[f"{metric_name}_mean"] = float(values.mean())
                item[f"{metric_name}_std"] = float(values.std(ddof=0))
        summary.append(item)
    return summary


def maybe_plot(summary_rows: list[dict[str, Any]], save_dir: Path, kl_metric_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    metrics = [
        ("refusal_mean_direction_projection_mean", "Mean Refusal Direction Projection"),
        ("refusal_subspace_projection_norm_mean", "Refusal Subspace Projection Norm"),
        ("refusal_subspace_fraction_mean", "Refusal Subspace Fraction"),
        ("refusal_subspace_residual_norm_mean", "Refusal Subspace Residual Norm"),
        ("refusal_margin_mean", "Refusal Margin"),
    ]
    if any(row.get("harmfulness_probe_score_mean", "") != "" for row in summary_rows):
        metrics.insert(1, ("harmfulness_probe_score_mean", "Harmfulness Probe Score"))
    kl_summary_name = f"{kl_metric_name}_mean"
    if any(row.get(kl_summary_name, "") != "" for row in summary_rows):
        reference = kl_metric_name.removeprefix("kl_to_").replace("_", " ").title()
        metrics.insert(1, (kl_summary_name, f"KL to {reference} Distribution"))
    layers = sorted({row["layer"] for row in summary_rows})
    if not layers:
        return
    # Plot the final transformer layer by default; embeddings are layer 0.
    layer = max(layers)
    rows = [row for row in summary_rows if row["layer"] == layer]
    conditions = sorted({row["condition"] for row in rows})
    ks = sorted({row["k"] for row in rows})
    response_steps = sorted({row["response_step"] for row in rows})

    for metric_name, ylabel in metrics:
        for response_step in response_steps:
            plt.figure(figsize=(7, 4))
            for condition in conditions:
                ys = []
                xs = []
                for k in ks:
                    match = [
                        row
                        for row in rows
                        if row["condition"] == condition and row["k"] == k and row["response_step"] == response_step
                    ]
                    if not match or match[0].get(metric_name) in ("", None):
                        continue
                    xs.append(k)
                    ys.append(float(match[0][metric_name]))
                if xs:
                    plt.plot(xs, ys, marker="o", label=condition)
            if not plt.gca().lines:
                plt.close()
                continue
            plt.xlabel("Prefix length k")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} (layer {layer}, response step {response_step})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_dir / f"{metric_name}_layer{layer}_response_step{response_step}.png", dpi=200)
            plt.close()

        for k in ks:
            plt.figure(figsize=(7, 4))
            for condition in conditions:
                ys = []
                xs = []
                for response_step in response_steps:
                    match = [
                        row
                        for row in rows
                        if row["condition"] == condition and row["k"] == k and row["response_step"] == response_step
                    ]
                    if not match or match[0].get(metric_name) in ("", None):
                        continue
                    xs.append(response_step)
                    ys.append(float(match[0][metric_name]))
                if xs:
                    plt.plot(xs, ys, marker="o", label=condition)
            if not plt.gca().lines:
                plt.close()
                continue
            plt.xlabel("Generated tokens after prefill")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} (layer {layer}, k={k})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_dir / f"{metric_name}_layer{layer}_k{k}_by_response_step.png", dpi=200)
            plt.close()


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()

    if model_config.model_name_or_path is None:
        raise ValueError(
            "No model specified. Please provide --model_name_or_path (e.g., --model_name_or_path ckpts/meta-llama/Llama-3.2-3B)"
        )

    if args.eval_template not in common_eval_template:
        raise ValueError(f"eval_template {args.eval_template} not maintained")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    ks = parse_int_list(args.ks)
    response_steps = sorted(set(parse_int_list(args.response_steps)))
    if any(response_step < 0 for response_step in response_steps):
        raise ValueError("--response_steps must be non-negative")
    layer_selector = parse_layer_selector(args.layers)
    target_device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if model_config.torch_dtype in ["auto", None]:
        torch_dtype = torch.float16 if target_device.type == "mps" and args.prefer_mps_float16 else model_config.torch_dtype
    else:
        torch_dtype = getattr(torch, model_config.torch_dtype)
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model, tokenizer = get_model.get_model(
        model_config.model_name_or_path,
        model_kwargs,
        model_family=args.model_family,
        padding_side="left",
    )
    model.eval()
    if quantization_config is None:
        model.to(target_device)
    model_device = next(getattr(model, "module", model).parameters()).device
    print(f"Using device={model_device}, torch_dtype={torch_dtype}, layers={args.layers}, compute_kl={args.compute_kl}")

    eval_template = common_eval_template[args.eval_template]
    system_prompt = eval_template["system_prompt"]
    input_template = eval_template["input_template"]
    generator = chat.Chat(model=model, prompt_style=args.prompt_style, tokenizer=tokenizer, init_system_prompt=system_prompt)

    adaptive_records = load_adaptive_records(args.adaptive_prefix_path, args.max_examples, args.seed)
    harmful_map = load_pair_map(args.harmful_prefix_path)
    refusal_map = load_pair_map(args.refusal_prefix_path) if Path(args.refusal_prefix_path).exists() else {}
    matched_records = [record for record in adaptive_records if record["instruction"] in harmful_map]
    if not matched_records:
        raise ValueError("No adaptive records matched instructions in the harmful prefix dataset")
    rng.shuffle(matched_records)

    refusal_token_ids = first_token_ids(tokenizer, args.refusal_token_strings)
    compliance_token_ids = first_token_ids(tokenizer, args.compliance_token_strings)

    probe_states: dict[int, dict[str, np.ndarray]] = {}
    probe_metric_rows: list[dict[str, Any]] = []
    conditions = ["clean", "refusal", "static", "adaptive"]
    if args.kl_reference_condition not in conditions:
        raise ValueError(f"--kl_reference_condition must be one of {conditions}")
    if args.baseline_condition not in conditions:
        raise ValueError(f"--baseline_condition must be one of {conditions}")
    principal_angle_conditions = parse_condition_list(args.principal_angle_conditions, conditions)
    kl_metric_name = f"kl_to_{args.kl_reference_condition}"
    if args.train_harmfulness_probe:
        harmful_prompts = load_harmful_prompts(args.harmful_prefix_path, args.max_probe_samples_per_class, args.seed)
        benign_prompts = load_benign_prompts(args.benign_path, args.max_probe_samples_per_class, args.seed)
        probe_examples = [(prompt, 1) for prompt in harmful_prompts] + [(prompt, 0) for prompt in benign_prompts]
        rng.shuffle(probe_examples)
        probe_prompts = []
        probe_positions = []
        probe_labels = []
        for instruction, label in probe_examples:
            messages = build_messages(system_prompt, input_template, instruction, "")
            prompt = render_messages(generator, messages)
            probe_prompts.append(prompt)
            probe_positions.append(find_last_user_token(prompt, instruction, tokenizer, args.max_length))
            probe_labels.append(label)
        probe_features, _, _, _, probe_layer_ids = forward_selected(
            model,
            tokenizer,
            probe_prompts,
            probe_positions,
            args.batch_size,
            args.max_length,
            layer_selector,
            args.use_cache,
            "Collecting probe training hidden states",
            collect_logprobs=False,
        )
        probe_features = probe_features[:, 0, :, :]
        labels = np.array(probe_labels, dtype=np.int64)
        train_indices, eval_indices = stratified_split(labels, args.probe_train_fraction, args.seed)
        for layer_array_idx, layer_idx in tqdm(
            list(enumerate(probe_layer_ids)),
            desc="Training harmfulness probes",
        ):
            metrics, state = train_probe(
                probe_features[train_indices, layer_array_idx, :],
                labels[train_indices],
                probe_features[eval_indices, layer_array_idx, :],
                labels[eval_indices],
                args,
                args.seed + layer_idx,
            )
            probe_states[layer_idx] = state
            probe_metric_rows.append({"layer": layer_idx, **metrics})
        save_csv(probe_metric_rows, save_dir / "probe_metrics.csv")

    def build_metadata(completed_ks: list[int], status: str, layer_ids: list[int] | None = None) -> dict[str, Any]:
        return {
            "args": asdict(args),
            "model_name_or_path": model_config.model_name_or_path,
            "resolved_device": str(model_device),
            "resolved_torch_dtype": str(torch_dtype),
            "layer_selector": args.layers,
            "kl_metric_name": kl_metric_name if args.compute_kl else None,
            "num_adaptive_records_loaded": len(adaptive_records),
            "num_matched_records": len(matched_records),
            "conditions": conditions,
            "ks": ks,
            "completed_ks": completed_ks,
            "status": status,
            "response_steps": response_steps,
            "layer_ids": layer_ids or [],
            "refusal_token_ids": refusal_token_ids,
            "compliance_token_ids": compliance_token_ids,
            "outputs": {
                "trajectory_per_example": str(save_dir / "trajectory_per_example.csv"),
                "trajectory_summary": str(save_dir / "trajectory_summary.csv"),
                "per_k_trajectory_per_example": str(save_dir / "trajectory_per_example_k{k}.csv"),
                "per_k_trajectory_summary": str(save_dir / "trajectory_summary_k{k}.csv"),
                "subspace_angles": str(save_dir / "subspace_angles.csv") if args.compute_principal_angles else None,
                "per_k_subspace_angles": str(save_dir / "subspace_angles_k{k}.csv") if args.compute_principal_angles else None,
                "probe_metrics": str(save_dir / "probe_metrics.csv") if args.train_harmfulness_probe else None,
            },
        }

    per_example_rows: list[dict[str, Any]] = []
    subspace_angle_rows: list[dict[str, Any]] = []
    completed_ks: list[int] = []
    last_selected_layer_ids: list[int] = []
    save_json(build_metadata(completed_ks, "started"), save_dir / "metadata.progress.json")
    for k in ks:
        k_per_example_rows: list[dict[str, Any]] = []
        k_subspace_angle_rows: list[dict[str, Any]] = []
        rendered: dict[str, list[str]] = {condition: [] for condition in conditions}
        positions: dict[str, list[list[int]]] = {condition: [] for condition in conditions}
        response_contexts: dict[str, list[list[int]]] = {condition: [] for condition in conditions}

        for record in matched_records:
            instruction = record["instruction"]
            static_prefix = first_k_tokens(tokenizer, harmful_map[instruction], k)
            adaptive_prefix = first_k_tokens(tokenizer, record["prefix"], k)
            refusal_source = refusal_map.get(instruction, args.refusal_fallback)
            refusal_prefix = first_k_tokens(tokenizer, refusal_source, k)
            prefixes = {
                "clean": "",
                "refusal": refusal_prefix,
                "static": static_prefix,
                "adaptive": adaptive_prefix,
            }
            for condition, prefix in prefixes.items():
                messages = build_messages(system_prompt, input_template, instruction, prefix)
                prompt = render_messages(generator, messages)
                rendered[condition].append(prompt)

        for condition in conditions:
            contexts, condition_positions = generate_greedy_contexts(
                model,
                tokenizer,
                rendered[condition],
                response_steps,
                args.batch_size,
                args.max_length,
                f"k={k} {condition} greedy response trajectory",
            )
            response_contexts[condition] = contexts
            positions[condition] = condition_positions

        features_by_condition: dict[str, np.ndarray] = {}
        logprobs_by_condition: dict[str, np.ndarray] = {}
        refusal_logmass_by_condition: dict[str, np.ndarray] = {}
        compliance_logmass_by_condition: dict[str, np.ndarray] = {}
        selected_layer_ids: list[int] | None = None
        for condition in conditions:
            features, logprobs, refusal_logmass, compliance_logmass, condition_layer_ids = forward_selected(
                model,
                tokenizer,
                response_contexts[condition],
                positions[condition],
                args.batch_size,
                args.max_length,
                layer_selector,
                args.use_cache,
                f"k={k} {condition}",
                collect_logprobs=args.compute_kl,
                refusal_token_ids=refusal_token_ids,
                compliance_token_ids=compliance_token_ids,
            )
            if selected_layer_ids is None:
                selected_layer_ids = condition_layer_ids
            elif condition_layer_ids != selected_layer_ids:
                raise RuntimeError("Layer selection differed across conditions")
            features_by_condition[condition] = features
            if logprobs is not None:
                logprobs_by_condition[condition] = logprobs
            if refusal_logmass is not None:
                refusal_logmass_by_condition[condition] = refusal_logmass
            if compliance_logmass is not None:
                compliance_logmass_by_condition[condition] = compliance_logmass

        if selected_layer_ids is None:
            selected_layer_ids = []
        last_selected_layer_ids = selected_layer_ids
        baseline_features = features_by_condition[args.baseline_condition]
        refusal_features = features_by_condition["refusal"]
        step_index = {response_step: idx for idx, response_step in enumerate(response_steps)}

        for response_step in response_steps:
            position_idx = step_index[response_step]
            baseline_at = baseline_features[:, position_idx, :, :]
            refusal_at = refusal_features[:, position_idx, :, :]
            refusal_deltas = refusal_at - baseline_at  # [n, layers, hidden_dim]

            # Build a multidimensional refusal subspace per layer from baseline->refusal deltas.
            # With the default clean baseline, this is a clean-to-refusal PCA/SVD subspace.
            refusal_mean_directions = refusal_deltas.mean(axis=0)  # [layers, hidden_dim]
            refusal_mean_norms = np.clip(
                np.linalg.norm(refusal_mean_directions, axis=-1, keepdims=True),
                1e-8,
                None,
            )
            unit_refusal_mean_directions = refusal_mean_directions / refusal_mean_norms

            refusal_bases: dict[int, np.ndarray] = {}
            refusal_basis_explained: dict[int, float] = {}
            for layer_array_idx, layer_idx in enumerate(selected_layer_ids):
                basis, _, explained = orthonormal_basis_from_deltas(
                    refusal_deltas[:, layer_array_idx, :],
                    args.refusal_subspace_rank,
                    center=args.center_subspace_deltas,
                )
                refusal_bases[layer_array_idx] = basis
                refusal_basis_explained[layer_array_idx] = explained

            # Principal-angle analysis between low-rank condition-specific delta subspaces.
            if args.compute_principal_angles:
                for layer_array_idx, layer_idx in enumerate(selected_layer_ids):
                    condition_bases: dict[str, np.ndarray] = {}
                    condition_explained: dict[str, float] = {}
                    condition_ranks: dict[str, int] = {}
                    for angle_condition in principal_angle_conditions:
                        condition_at = features_by_condition[angle_condition][:, position_idx, :, :]
                        condition_deltas = condition_at[:, layer_array_idx, :] - baseline_at[:, layer_array_idx, :]
                        basis, _, explained = orthonormal_basis_from_deltas(
                            condition_deltas,
                            args.condition_subspace_rank,
                            center=args.center_subspace_deltas,
                        )
                        condition_bases[angle_condition] = basis
                        condition_explained[angle_condition] = explained
                        condition_ranks[angle_condition] = basis.shape[1]

                    for left_idx, condition_a in enumerate(principal_angle_conditions):
                        for condition_b in principal_angle_conditions[left_idx + 1 :]:
                            angles = principal_angles_degrees(
                                condition_bases[condition_a],
                                condition_bases[condition_b],
                            )
                            row: dict[str, Any] = {
                                "k": k,
                                "response_step": response_step,
                                "layer": layer_idx,
                                "baseline_condition": args.baseline_condition,
                                "condition_a": condition_a,
                                "condition_b": condition_b,
                                "rank_a": condition_ranks[condition_a],
                                "rank_b": condition_ranks[condition_b],
                                "explained_variance_a": condition_explained[condition_a],
                                "explained_variance_b": condition_explained[condition_b],
                                "num_angles": int(len(angles)),
                            }
                            if len(angles) == 0:
                                row.update(
                                    {
                                        "principal_angle_min_deg": "",
                                        "principal_angle_mean_deg": "",
                                        "principal_angle_max_deg": "",
                                        "subspace_alignment_mean_cos2": "",
                                    }
                                )
                            else:
                                cos2 = np.square(np.cos(np.radians(angles)))
                                row.update(
                                    {
                                        "principal_angle_min_deg": float(np.min(angles)),
                                        "principal_angle_mean_deg": float(np.mean(angles)),
                                        "principal_angle_max_deg": float(np.max(angles)),
                                        "subspace_alignment_mean_cos2": float(np.mean(cos2)),
                                    }
                                )
                                for angle_idx, angle in enumerate(angles[: min(8, len(angles))], start=1):
                                    row[f"principal_angle_{angle_idx}_deg"] = float(angle)
                            k_subspace_angle_rows.append(row)

            reference_logprobs_at = None
            if args.compute_kl:
                reference_logprobs_at = logprobs_by_condition[args.kl_reference_condition][
                    :, position_idx, :
                ]

            for condition in conditions:
                features_at = features_by_condition[condition][:, position_idx, :, :]
                deltas_from_baseline = features_at - baseline_at
                delta_norm_from_baseline = np.linalg.norm(deltas_from_baseline, axis=-1)
                refusal_mean_direction_projection = np.sum(
                    deltas_from_baseline * unit_refusal_mean_directions[None, :, :],
                    axis=-1,
                )
                # Backwards-compatible alias for older plotting/CSV consumers.
                safety_projection = refusal_mean_direction_projection

                if args.compute_kl:
                    condition_logprobs = logprobs_by_condition[condition][:, position_idx, :]
                    condition_probs = np.exp(condition_logprobs)
                    kl_to_reference = np.sum(
                        condition_probs * (condition_logprobs - reference_logprobs_at),
                        axis=-1,
                    )
                    refusal_logmass = (
                        np.logaddexp.reduce(condition_logprobs[:, refusal_token_ids], axis=-1)
                        if refusal_token_ids
                        else np.full(len(matched_records), np.nan)
                    )
                    compliance_logmass = (
                        np.logaddexp.reduce(condition_logprobs[:, compliance_token_ids], axis=-1)
                        if compliance_token_ids
                        else np.full(len(matched_records), np.nan)
                    )
                else:
                    kl_to_reference = None
                    refusal_logmass = refusal_logmass_by_condition.get(
                        condition,
                        np.full((len(matched_records), len(response_steps)), np.nan),
                    )[:, position_idx]
                    compliance_logmass = compliance_logmass_by_condition.get(
                        condition,
                        np.full((len(matched_records), len(response_steps)), np.nan),
                    )[:, position_idx]
                refusal_margin = refusal_logmass - compliance_logmass

                for layer_array_idx, layer_idx in enumerate(selected_layer_ids):
                    layer_deltas = deltas_from_baseline[:, layer_array_idx, :]
                    layer_refusal_deltas = refusal_deltas[:, layer_array_idx, :]
                    layer_refusal_mean = np.repeat(
                        refusal_mean_directions[layer_array_idx : layer_array_idx + 1, :],
                        repeats=layer_deltas.shape[0],
                        axis=0,
                    )
                    projection_norm, projection_fraction, residual_norm = project_metrics_for_basis(
                        layer_deltas,
                        refusal_bases[layer_array_idx],
                    )
                    cosine_to_paired_refusal = cosine_to_reference(layer_deltas, layer_refusal_deltas)
                    cosine_to_refusal_mean = cosine_to_reference(layer_deltas, layer_refusal_mean)
                    subspace_rank_used = refusal_bases[layer_array_idx].shape[1]
                    subspace_explained = refusal_basis_explained[layer_array_idx]

                    for example_idx, record in enumerate(matched_records):
                        row = {
                            "k": k,
                            "response_step": response_step,
                            "condition": condition,
                            "layer": layer_idx,
                            "example_idx": example_idx,
                            "record_id": record.get("id", example_idx),
                            "instruction": record["instruction"],
                            "safety_projection": float(safety_projection[example_idx, layer_array_idx]),
                            "delta_norm_from_baseline": float(delta_norm_from_baseline[example_idx, layer_array_idx]),
                            "refusal_mean_direction_projection": float(refusal_mean_direction_projection[example_idx, layer_array_idx]),
                            "refusal_subspace_projection_norm": float(projection_norm[example_idx]),
                            "refusal_subspace_fraction": float(projection_fraction[example_idx]),
                            "refusal_subspace_residual_norm": float(residual_norm[example_idx]),
                            "cosine_to_refusal_delta": float(cosine_to_paired_refusal[example_idx]),
                            "cosine_to_refusal_mean_direction": float(cosine_to_refusal_mean[example_idx]),
                            "refusal_subspace_rank_used": int(subspace_rank_used),
                            "refusal_subspace_explained_variance": float(subspace_explained),
                            "refusal_logmass": float(refusal_logmass[example_idx]),
                            "compliance_logmass": float(compliance_logmass[example_idx]),
                            "refusal_margin": float(refusal_margin[example_idx]),
                        }
                        if kl_to_reference is not None:
                            row[kl_metric_name] = float(kl_to_reference[example_idx])
                        if layer_idx in probe_states:
                            probe_score = apply_probe(
                                features_at[example_idx : example_idx + 1, layer_array_idx, :],
                                probe_states[layer_idx],
                            )[0]
                            row["harmfulness_probe_score"] = float(probe_score)
                        k_per_example_rows.append(row)

        k_summary_rows = aggregate_rows(
            k_per_example_rows,
            include_probe=args.train_harmfulness_probe,
            include_kl=args.compute_kl,
            kl_metric_name=kl_metric_name,
        )
        save_csv(k_per_example_rows, save_dir / f"trajectory_per_example_k{k}.csv")
        save_csv(k_summary_rows, save_dir / f"trajectory_summary_k{k}.csv")
        if args.compute_principal_angles:
            save_csv(k_subspace_angle_rows, save_dir / f"subspace_angles_k{k}.csv")

        per_example_rows.extend(k_per_example_rows)
        if args.compute_principal_angles:
            subspace_angle_rows.extend(k_subspace_angle_rows)
            save_csv(subspace_angle_rows, save_dir / "subspace_angles.csv")
        summary_rows = aggregate_rows(
            per_example_rows,
            include_probe=args.train_harmfulness_probe,
            include_kl=args.compute_kl,
            kl_metric_name=kl_metric_name,
        )
        save_csv(per_example_rows, save_dir / "trajectory_per_example.csv")
        save_csv(summary_rows, save_dir / "trajectory_summary.csv")
        completed_ks.append(k)
        save_json(
            build_metadata(completed_ks, "running", last_selected_layer_ids),
            save_dir / "metadata.progress.json",
        )
        print(f"Saved partial trajectory analysis for k={k} to {save_dir}")

    summary_rows = aggregate_rows(
        per_example_rows,
        include_probe=args.train_harmfulness_probe,
        include_kl=args.compute_kl,
        kl_metric_name=kl_metric_name,
    )
    save_csv(per_example_rows, save_dir / "trajectory_per_example.csv")
    save_csv(summary_rows, save_dir / "trajectory_summary.csv")
    if args.compute_principal_angles:
        save_csv(subspace_angle_rows, save_dir / "subspace_angles.csv")
    maybe_plot(summary_rows, save_dir, kl_metric_name)

    metadata = build_metadata(completed_ks, "completed", last_selected_layer_ids)
    save_json(metadata, save_dir / "metadata.json")
    save_json(metadata, save_dir / "metadata.progress.json")

    print(f"Saved trajectory analysis to {save_dir}")
    print(f"Matched prompts: {len(matched_records)}")


if __name__ == "__main__":
    main()
