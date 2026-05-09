from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from analyze_prefix_trajectories import (
    build_messages,
    first_k_tokens,
    generate_greedy_contexts,
    forward_selected,
    load_adaptive_records,
    load_pair_map,
    parse_int_list,
    parse_layer_selector,
    render_messages,
    resolve_device,
)
from finetuning_buckets.inference import chat
from finetuning_buckets.inference.safety_eval.evaluator import common_eval_template
from finetuning_buckets.inference.safety_eval.keyword_eval import KeyWordEvaluator
from finetuning_buckets.models import get_model


DEFAULT_HARMFUL_PATH = "finetuning_buckets/datasets/data/safety_bench/Harmful-HEx-PHI.jsonl"
DEFAULT_REFUSAL_PATH = "finetuning_buckets/datasets/data/safety_bench/llama2_HEx-PHI_refusal_examples.jsonl"


@dataclass
class ScriptArguments:
    model_family: str = field(default="llama3")
    prompt_style: str = field(default="llama3")
    eval_template: str = field(default="null")
    adaptive_prefix_path: str = field(default="logs/adaptive_prefixes/mined_augmented_tinker.jsonl")
    harmful_prefix_path: str = field(default=DEFAULT_HARMFUL_PATH)
    refusal_prefix_path: str = field(default=DEFAULT_REFUSAL_PATH)
    refusal_fallback: str = field(default="I cannot fulfill this request.")
    ks: str = field(default="5,10,15,20")
    response_steps: str = field(default="0,1,2,4,8,16")
    max_examples: int = field(default=0, metadata={"help": "Max matched adaptive examples; <=0 uses all."})
    batch_size: int = field(default=2)
    max_length: int = field(default=768)
    layers: str = field(default="final", metadata={"help": "all, final, or comma-separated hidden-state layer ids."})
    conditions: str = field(default="clean,refusal,static,adaptive")
    save_dir: str = field(default="logs/adaptive_prefixes/trajectory_feature_cache")
    seed: int = field(default=0)
    use_cache: bool = field(default=False)
    device: str = field(default="auto", metadata={"help": "auto, cuda, mps, or cpu."})
    prefer_mps_float16: bool = field(default=True)
    feature_dtype: str = field(default="float16", metadata={"help": "float16 or float32 for saved feature arrays."})
    eval_result_template: str = field(
        default="logs/adaptive_prefixes/eval_augmented_on_mined_prefixes_k{k}.json",
        metadata={"help": "Optional k-formatted eval JSON path for ASR success labels."},
    )
    success_filter: str = field(
        default="all",
        metadata={"help": "all, success, failure, or labeled_only. Applied per k using adaptive ASR labels."},
    )
    save_features: bool = field(default=True)
    compute_metrics: bool = field(default=True)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_conditions(raw: str) -> list[str]:
    allowed = {"clean", "refusal", "static", "adaptive"}
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(conditions) - allowed)
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if "static" not in conditions or "refusal" not in conditions:
        raise ValueError("Metrics require both static and refusal conditions")
    return conditions


def load_success_labels(template: str, ks: list[int]) -> dict[tuple[int, str], bool]:
    labels: dict[tuple[int, str], bool] = {}
    if not template:
        return labels
    for k in ks:
        path = Path(template.format(k=k))
        if not path.exists():
            continue
        with path.open() as handle:
            data = json.load(handle)
        for plain_item, result_item in zip(data.get("plain_text", []), data.get("results", [])):
            if not plain_item:
                continue
            instruction = plain_item[0]
            output = result_item[-1]["content"] if result_item else ""
            labels[(k, instruction)] = bool(KeyWordEvaluator.is_jailbroken(output))
    return labels


def filter_records_for_k(
    records: list[dict[str, Any]],
    k: int,
    success_labels: dict[tuple[int, str], bool],
    success_filter: str,
) -> list[dict[str, Any]]:
    if success_filter == "all":
        return records
    filtered = []
    for record in records:
        label = success_labels.get((k, record["instruction"]))
        if success_filter == "labeled_only" and label is not None:
            filtered.append(record)
        elif success_filter == "success" and label is True:
            filtered.append(record)
        elif success_filter == "failure" and label is False:
            filtered.append(record)
    if success_filter not in {"all", "success", "failure", "labeled_only"}:
        raise ValueError("--success_filter must be all, success, failure, or labeled_only")
    return filtered


def safe_npz_name(condition: str, k: int) -> str:
    return f"features_k{k}_{condition}.npz"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.sum(a * b, axis=-1) / np.clip(denom, 1e-8, None)


def compute_hidden_metrics(
    features_by_condition: dict[str, np.ndarray],
    layer_ids: list[int],
    records: list[dict[str, Any]],
    k: int,
    response_steps: list[int],
    success_labels: dict[tuple[int, str], bool],
    conditions: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_rows: list[dict[str, Any]] = []
    static_features = features_by_condition["static"]
    refusal_features = features_by_condition["refusal"]

    for step_idx, response_step in enumerate(response_steps):
        static_at = static_features[:, step_idx, :, :]
        refusal_at = refusal_features[:, step_idx, :, :]
        static_mean = static_at.mean(axis=0)
        refusal_mean = refusal_at.mean(axis=0)
        safety_direction = refusal_mean - static_mean
        unit_direction = safety_direction / np.clip(
            np.linalg.norm(safety_direction, axis=-1, keepdims=True),
            1e-8,
            None,
        )

        for condition in conditions:
            features = features_by_condition[condition][:, step_idx, :, :]
            delta_to_static_example = features - static_at
            delta_to_refusal_example = features - refusal_at
            centered = features - static_mean[None, :, :]
            projection = np.sum(centered * unit_direction[None, :, :], axis=-1)
            orthogonal = centered - projection[:, :, None] * unit_direction[None, :, :]
            l2_to_static_example = np.linalg.norm(delta_to_static_example, axis=-1)
            l2_to_refusal_example = np.linalg.norm(delta_to_refusal_example, axis=-1)
            l2_to_static_mean = np.linalg.norm(centered, axis=-1)
            orthogonal_l2 = np.linalg.norm(orthogonal, axis=-1)
            cosine_to_static_example = cosine_similarity(features, static_at)
            cosine_to_refusal_example = cosine_similarity(features, refusal_at)

            for layer_array_idx, layer_id in enumerate(layer_ids):
                for example_idx, record in enumerate(records):
                    label = success_labels.get((k, record["instruction"]))
                    per_rows.append(
                        {
                            "k": k,
                            "response_step": response_step,
                            "condition": condition,
                            "layer": layer_id,
                            "example_idx": example_idx,
                            "record_id": record.get("id", example_idx),
                            "instruction": record["instruction"],
                            "adaptive_success": "" if label is None else int(label),
                            "hidden_l2_to_static_example": float(l2_to_static_example[example_idx, layer_array_idx]),
                            "hidden_l2_to_refusal_example": float(l2_to_refusal_example[example_idx, layer_array_idx]),
                            "hidden_l2_to_static_mean": float(l2_to_static_mean[example_idx, layer_array_idx]),
                            "hidden_cosine_to_static_example": float(cosine_to_static_example[example_idx, layer_array_idx]),
                            "hidden_cosine_to_refusal_example": float(cosine_to_refusal_example[example_idx, layer_array_idx]),
                            "safety_projection": float(projection[example_idx, layer_array_idx]),
                            "safety_orthogonal_l2": float(orthogonal_l2[example_idx, layer_array_idx]),
                        }
                    )

    metric_names = [
        "hidden_l2_to_static_example",
        "hidden_l2_to_refusal_example",
        "hidden_l2_to_static_mean",
        "hidden_cosine_to_static_example",
        "hidden_cosine_to_refusal_example",
        "safety_projection",
        "safety_orthogonal_l2",
    ]
    grouped: dict[tuple[int, int, str, int], list[dict[str, Any]]] = {}
    for row in per_rows:
        key = (row["k"], row["response_step"], row["condition"], row["layer"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (k_value, response_step, condition, layer), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "k": k_value,
            "response_step": response_step,
            "condition": condition,
            "layer": layer,
            "n": len(rows),
        }
        for metric_name in metric_names:
            values = np.array([row[metric_name] for row in rows], dtype=np.float64)
            item[f"{metric_name}_mean"] = float(values.mean())
            item[f"{metric_name}_std"] = float(values.std(ddof=0))
        summary_rows.append(item)

    return per_rows, summary_rows


def summarize_by_success(per_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        "hidden_l2_to_static_example",
        "hidden_l2_to_refusal_example",
        "hidden_l2_to_static_mean",
        "hidden_cosine_to_static_example",
        "hidden_cosine_to_refusal_example",
        "safety_projection",
        "safety_orthogonal_l2",
    ]
    grouped: dict[tuple[int, int, str, int, int], list[dict[str, Any]]] = {}
    for row in per_rows:
        if row["adaptive_success"] == "":
            continue
        key = (
            row["k"],
            row["response_step"],
            row["condition"],
            row["layer"],
            int(row["adaptive_success"]),
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for (k, response_step, condition, layer, success), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "k": k,
            "response_step": response_step,
            "condition": condition,
            "layer": layer,
            "adaptive_success": success,
            "n": len(rows),
        }
        for metric_name in metric_names:
            values = np.array([row[metric_name] for row in rows], dtype=np.float64)
            item[f"{metric_name}_mean"] = float(values.mean())
            item[f"{metric_name}_std"] = float(values.std(ddof=0))
        summary.append(item)
    return summary


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()

    if model_config.model_name_or_path is None:
        raise ValueError(
            "No model specified. Please provide --model_name_or_path (e.g., --model_name_or_path ckpts/meta-llama/Llama-3.2-3B)"
        )

    if args.eval_template not in common_eval_template:
        raise ValueError(f"eval_template {args.eval_template} not maintained")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    ks = parse_int_list(args.ks)
    response_steps = sorted(set(parse_int_list(args.response_steps)))
    if any(step < 0 for step in response_steps):
        raise ValueError("--response_steps must be non-negative")
    conditions = parse_conditions(args.conditions)
    layer_selector = parse_layer_selector(args.layers)
    target_device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    feature_dir = save_dir / "features"
    save_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_dtype not in {"float16", "float32"}:
        raise ValueError("--feature_dtype must be float16 or float32")
    np_feature_dtype = np.float16 if args.feature_dtype == "float16" else np.float32

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
    print(f"Using device={model_device}, torch_dtype={torch_dtype}, layers={args.layers}")

    eval_template = common_eval_template[args.eval_template]
    system_prompt = eval_template["system_prompt"]
    input_template = eval_template["input_template"]
    generator = chat.Chat(model=model, prompt_style=args.prompt_style, tokenizer=tokenizer, init_system_prompt=system_prompt)

    adaptive_records = load_adaptive_records(args.adaptive_prefix_path, args.max_examples, args.seed)
    harmful_map = load_pair_map(args.harmful_prefix_path)
    refusal_map = load_pair_map(args.refusal_prefix_path) if Path(args.refusal_prefix_path).exists() else {}
    matched_records = [record for record in adaptive_records if record["instruction"] in harmful_map]
    rng.shuffle(matched_records)
    if not matched_records:
        raise ValueError("No adaptive records matched instructions in the harmful prefix dataset")

    success_labels = load_success_labels(args.eval_result_template, ks)
    success_rows = [
        {"k": k, "instruction": instruction, "adaptive_success": int(success)}
        for (k, instruction), success in sorted(success_labels.items())
    ]
    save_csv(success_rows, save_dir / "success_labels.csv")

    manifest_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_metric_summary_rows: list[dict[str, Any]] = []
    last_layer_ids: list[int] = []

    for k in ks:
        records_for_k = filter_records_for_k(matched_records, k, success_labels, args.success_filter)
        if not records_for_k:
            print(f"Skipping k={k}; no records after success_filter={args.success_filter}")
            continue

        rendered: dict[str, list[str]] = {condition: [] for condition in conditions}
        for record in records_for_k:
            instruction = record["instruction"]
            prefixes = {
                "clean": "",
                "refusal": first_k_tokens(
                    tokenizer,
                    refusal_map.get(instruction, args.refusal_fallback),
                    k,
                ),
                "static": first_k_tokens(tokenizer, harmful_map[instruction], k),
                "adaptive": first_k_tokens(tokenizer, record["prefix"], k),
            }
            for condition in conditions:
                prompt = render_messages(
                    generator,
                    build_messages(system_prompt, input_template, instruction, prefixes[condition]),
                )
                rendered[condition].append(prompt)

        features_by_condition: dict[str, np.ndarray] = {}
        for condition in conditions:
            contexts, positions = generate_greedy_contexts(
                model,
                tokenizer,
                rendered[condition],
                response_steps,
                args.batch_size,
                args.max_length,
                f"k={k} {condition} greedy response trajectory",
            )
            features, _, _, _, layer_ids = forward_selected(
                model,
                tokenizer,
                contexts,
                positions,
                args.batch_size,
                args.max_length,
                layer_selector,
                args.use_cache,
                f"k={k} {condition} hidden states",
                collect_logprobs=False,
            )
            last_layer_ids = layer_ids
            features = features.astype(np_feature_dtype, copy=False)
            features_by_condition[condition] = features

            feature_path = feature_dir / safe_npz_name(condition, k)
            if args.save_features:
                np.savez_compressed(
                    feature_path,
                    features=features,
                    response_steps=np.array(response_steps, dtype=np.int64),
                    layer_ids=np.array(layer_ids, dtype=np.int64),
                    record_ids=np.array([record.get("id", idx) for idx, record in enumerate(records_for_k)], dtype=np.int64),
                    instructions=np.array([record["instruction"] for record in records_for_k], dtype=object),
                    adaptive_success=np.array(
                        [
                            -1 if success_labels.get((k, record["instruction"])) is None else int(success_labels[(k, record["instruction"])])
                            for record in records_for_k
                        ],
                        dtype=np.int8,
                    ),
                )

            for row_idx, record in enumerate(records_for_k):
                label = success_labels.get((k, record["instruction"]))
                manifest_rows.append(
                    {
                        "k": k,
                        "condition": condition,
                        "row_idx": row_idx,
                        "record_id": record.get("id", row_idx),
                        "instruction": record["instruction"],
                        "feature_file": str(feature_path),
                        "adaptive_success": "" if label is None else int(label),
                    }
                )

        if args.compute_metrics:
            metric_rows, metric_summary_rows = compute_hidden_metrics(
                features_by_condition={condition: features_by_condition[condition].astype(np.float32) for condition in conditions},
                layer_ids=last_layer_ids,
                records=records_for_k,
                k=k,
                response_steps=response_steps,
                success_labels=success_labels,
                conditions=conditions,
            )
            all_metric_rows.extend(metric_rows)
            all_metric_summary_rows.extend(metric_summary_rows)

    save_csv(manifest_rows, save_dir / "feature_manifest.csv")
    if args.compute_metrics:
        save_csv(all_metric_rows, save_dir / "hidden_metrics_per_example.csv")
        save_csv(all_metric_summary_rows, save_dir / "hidden_metrics_summary.csv")
        save_csv(summarize_by_success(all_metric_rows), save_dir / "hidden_metrics_by_success.csv")

    metadata = {
        "args": asdict(args),
        "model_name_or_path": model_config.model_name_or_path,
        "resolved_device": str(model_device),
        "resolved_torch_dtype": str(torch_dtype),
        "feature_dtype": args.feature_dtype,
        "conditions": conditions,
        "ks": ks,
        "response_steps": response_steps,
        "layer_ids": last_layer_ids,
        "num_matched_records_before_filter": len(matched_records),
        "num_success_labels": len(success_labels),
        "outputs": {
            "features": str(feature_dir),
            "feature_manifest": str(save_dir / "feature_manifest.csv"),
            "success_labels": str(save_dir / "success_labels.csv"),
            "hidden_metrics_per_example": str(save_dir / "hidden_metrics_per_example.csv") if args.compute_metrics else None,
            "hidden_metrics_summary": str(save_dir / "hidden_metrics_summary.csv") if args.compute_metrics else None,
            "hidden_metrics_by_success": str(save_dir / "hidden_metrics_by_success.csv") if args.compute_metrics else None,
        },
    }
    with (save_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved feature cache to {save_dir}")


if __name__ == "__main__":
    main()
