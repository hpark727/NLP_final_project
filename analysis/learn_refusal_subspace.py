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
    ks: str = field(default="5,10,20,40")
    response_steps: str = field(default="0")
    max_examples: int = field(default=0, metadata={"help": "Max matched adaptive examples; <=0 uses all."})
    batch_size: int = field(default=2)
    max_length: int = field(default=768)
    layers: str = field(default="final")
    n_components: int = field(default=8)
    center_deltas: bool = field(
        default=False,
        metadata={"help": "Center refusal-static deltas before SVD. Default keeps the mean direction in PC1."},
    )
    save_features: bool = field(default=False)
    save_dir: str = field(default="logs/adaptive_prefixes/refusal_subspace")
    seed: int = field(default=0)
    use_cache: bool = field(default=False)
    device: str = field(default="auto", metadata={"help": "auto, cuda, mps, or cpu."})
    prefer_mps_float16: bool = field(default=True)
    eval_result_template: str = field(
        default="logs/adaptive_prefixes/eval_augmented_on_mined_prefixes_k{k}.json",
        metadata={"help": "Optional k-formatted adaptive eval JSON path for success labels."},
    )
    baseline_condition: str = field(
        default="static",
        metadata={"help": "The condition to use as the baseline for delta calculations (clean, refusal, static, or adaptive)."},
    )


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.sum(a * b, axis=-1) / np.clip(denom, 1e-8, None)


def fit_subspace(deltas: np.ndarray, n_components: int, center: bool) -> dict[str, np.ndarray]:
    mean_delta = deltas.mean(axis=0)
    mean_norm = np.linalg.norm(mean_delta)
    mean_unit = mean_delta / max(mean_norm, 1e-8)
    fit_matrix = deltas - mean_delta[None, :] if center else deltas
    _, singular_values, vt = np.linalg.svd(fit_matrix.astype(np.float32), full_matrices=False)
    total_energy = np.square(singular_values).sum()
    explained = np.square(singular_values) / max(total_energy, 1e-12)
    n_keep = min(n_components, vt.shape[0])
    return {
        "basis": vt[:n_keep].astype(np.float32),
        "singular_values": singular_values.astype(np.float32),
        "explained_energy_ratio": explained.astype(np.float32),
        "mean_delta": mean_delta.astype(np.float32),
        "mean_unit": mean_unit.astype(np.float32),
        "fit_center": np.array(center, dtype=np.bool_),
    }


def project_onto_basis(vectors: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = vectors @ basis.T
    projected = coefficients @ basis
    return coefficients, projected


def build_condition_prompts(
    records: list[dict[str, Any]],
    k: int,
    tokenizer,
    generator: chat.Chat,
    system_prompt: str | None,
    input_template: str | None,
    harmful_map: dict[str, str],
    refusal_map: dict[str, str],
    refusal_fallback: str,
) -> dict[str, list[str]]:
    rendered = {"clean": [], "refusal": [], "static": [], "adaptive": []}
    for record in records:
        instruction = record["instruction"]
        prefixes = {
            "clean": "",
            "refusal": first_k_tokens(tokenizer, refusal_map.get(instruction, refusal_fallback), k),
            "static": first_k_tokens(tokenizer, harmful_map[instruction], k),
            "adaptive": first_k_tokens(tokenizer, record["prefix"], k),
        }
        for condition, prefix in prefixes.items():
            rendered[condition].append(
                render_messages(
                    generator,
                    build_messages(system_prompt, input_template, instruction, prefix),
                )
            )
    return rendered


def collect_features_for_condition(
    model,
    tokenizer,
    prompts: list[str],
    response_steps: list[int],
    args: ScriptArguments,
    layer_selector: str | list[int],
    desc: str,
) -> tuple[np.ndarray, list[int]]:
    contexts, positions = generate_greedy_contexts(
        model,
        tokenizer,
        prompts,
        response_steps,
        args.batch_size,
        args.max_length,
        desc=f"{desc} greedy contexts",
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
        desc=f"{desc} hidden states",
        collect_logprobs=False,
    )
    return features.astype(np.float32, copy=False), layer_ids


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
    layer_selector = parse_layer_selector(args.layers)
    target_device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    component_dir = save_dir / "components"
    feature_dir = save_dir / "features"
    save_dir.mkdir(parents=True, exist_ok=True)
    component_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_dir.mkdir(parents=True, exist_ok=True)

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
    per_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    manifest_rows = [
        {
            "example_idx": idx,
            "record_id": record.get("id", idx),
            "instruction": record["instruction"],
        }
        for idx, record in enumerate(matched_records)
    ]
    save_csv(manifest_rows, save_dir / "example_manifest.csv")

    for k in ks:
        rendered = build_condition_prompts(
            matched_records,
            k,
            tokenizer,
            generator,
            system_prompt,
            input_template,
            harmful_map,
            refusal_map,
            args.refusal_fallback,
        )
        features_by_condition: dict[str, np.ndarray] = {}
        layer_ids: list[int] = []
        for condition, prompts in rendered.items():
            features, layer_ids = collect_features_for_condition(
                model,
                tokenizer,
                prompts,
                response_steps,
                args,
                layer_selector,
                desc=f"k={k} {condition}",
            )
            features_by_condition[condition] = features
            if args.save_features:
                np.savez_compressed(
                    feature_dir / f"features_k{k}_{condition}.npz",
                    features=features.astype(np.float16),
                    response_steps=np.array(response_steps, dtype=np.int64),
                    layer_ids=np.array(layer_ids, dtype=np.int64),
                    instructions=np.array([record["instruction"] for record in matched_records], dtype=object),
                )

        baseline_features = features_by_condition[args.baseline_condition]
        refusal_features = features_by_condition["refusal"]
        for step_idx, response_step in enumerate(response_steps):
            for layer_array_idx, layer_id in enumerate(layer_ids):
                baseline_at = baseline_features[:, step_idx, layer_array_idx, :]
                refusal_at = refusal_features[:, step_idx, layer_array_idx, :]
                refusal_deltas = refusal_at - baseline_at
                subspace = fit_subspace(refusal_deltas, args.n_components, args.center_deltas)
                basis = subspace["basis"]
                mean_unit = subspace["mean_unit"]
                component_path = component_dir / f"refusal_subspace_k{k}_step{response_step}_layer{layer_id}.npz"
                np.savez_compressed(
                    component_path,
                    **subspace,
                    k=np.array(k, dtype=np.int64),
                    response_step=np.array(response_step, dtype=np.int64),
                    layer=np.array(layer_id, dtype=np.int64),
                )

                for component_idx, (singular_value, explained) in enumerate(
                    zip(subspace["singular_values"], subspace["explained_energy_ratio"])
                ):
                    energy_rows.append(
                        {
                            "k": k,
                            "response_step": response_step,
                            "layer": layer_id,
                            "component": component_idx,
                            "singular_value": float(singular_value),
                            "explained_energy_ratio": float(explained),
                            "cumulative_energy_ratio": float(subspace["explained_energy_ratio"][: component_idx + 1].sum()),
                        }
                    )

                for condition, features in features_by_condition.items():
                    condition_at = features[:, step_idx, layer_array_idx, :]
                    vectors = condition_at - baseline_at
                    coefficients, projected = project_onto_basis(vectors, basis)
                    vector_norm = np.linalg.norm(vectors, axis=-1)
                    projection_norm = np.linalg.norm(projected, axis=-1)
                    residual_norm = np.linalg.norm(vectors - projected, axis=-1)
                    subspace_fraction = projection_norm / np.clip(vector_norm, 1e-8, None)
                    mean_projection = vectors @ mean_unit
                    cosine_to_refusal_delta = cosine(vectors, refusal_deltas)
                    for example_idx, record in enumerate(matched_records):
                        label = success_labels.get((k, record["instruction"]))
                        row = {
                            "k": k,
                            "response_step": response_step,
                            "layer": layer_id,
                            "condition": condition,
                            "example_idx": example_idx,
                            "record_id": record.get("id", example_idx),
                            "instruction": record["instruction"],
                            "adaptive_success": "" if label is None else int(label),
                            "delta_norm_from_baseline": float(vector_norm[example_idx]),
                            "refusal_mean_direction_projection": float(mean_projection[example_idx]),
                            "refusal_subspace_projection_norm": float(projection_norm[example_idx]),
                            "refusal_subspace_fraction": float(subspace_fraction[example_idx]),
                            "refusal_subspace_residual_norm": float(residual_norm[example_idx]),
                            "cosine_to_refusal_delta": float(cosine_to_refusal_delta[example_idx]),
                        }
                        for component_idx in range(basis.shape[0]):
                            row[f"pc{component_idx + 1}_coef"] = float(coefficients[example_idx, component_idx])
                        per_rows.append(row)

                for condition in features_by_condition:
                    condition_rows = [
                        row
                        for row in per_rows
                        if row["k"] == k
                        and row["response_step"] == response_step
                        and row["layer"] == layer_id
                        and row["condition"] == condition
                    ]
                    item: dict[str, Any] = {
                        "k": k,
                        "response_step": response_step,
                        "layer": layer_id,
                        "condition": condition,
                        "n": len(condition_rows),
                    }
                    for metric_name in [
                        "delta_norm_from_baseline",
                        "refusal_mean_direction_projection",
                        "refusal_subspace_projection_norm",
                        "refusal_subspace_fraction",
                        "refusal_subspace_residual_norm",
                        "cosine_to_refusal_delta",
                    ]:
                        values = np.array([row[metric_name] for row in condition_rows], dtype=np.float64)
                        item[f"{metric_name}_mean"] = float(values.mean())
                        item[f"{metric_name}_std"] = float(values.std(ddof=0))
                    summary_rows.append(item)

    save_csv(per_rows, save_dir / "refusal_subspace_per_example.csv")
    save_csv(summary_rows, save_dir / "refusal_subspace_summary.csv")
    save_csv(energy_rows, save_dir / "refusal_subspace_energy.csv")

    metadata = {
        "args": asdict(args),
        "model_name_or_path": model_config.model_name_or_path,
        "resolved_device": str(model_device),
        "resolved_torch_dtype": str(torch_dtype),
        "num_examples": len(matched_records),
        "ks": ks,
        "response_steps": response_steps,
        "layer_ids": layer_ids,
        "outputs": {
            "components": str(component_dir),
            "per_example": str(save_dir / "refusal_subspace_per_example.csv"),
            "summary": str(save_dir / "refusal_subspace_summary.csv"),
            "energy": str(save_dir / "refusal_subspace_energy.csv"),
            "example_manifest": str(save_dir / "example_manifest.csv"),
        },
    }
    with (save_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved refusal subspace analysis to {save_dir}")


if __name__ == "__main__":
    main()
