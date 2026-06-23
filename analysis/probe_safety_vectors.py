from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.state import PartialState
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from finetuning_buckets.datasets.utils import get_eval_data
from finetuning_buckets.inference import chat
from finetuning_buckets.inference.safety_eval.evaluator import common_eval_template
from finetuning_buckets.models import get_model


@dataclass
class ScriptArguments:
    model_family: str = field(default="llama2", metadata={"help": "the model family"})
    prompt_style: str = field(default="llama2", metadata={"help": "the string prompt style"})
    eval_template: str = field(default="plain", metadata={"help": "the eval template"})
    conditions: str = field(
        default="plain,refusal_prefix,harmful_prefix",
        metadata={
            "help": (
                "Comma-separated prompt conditions. Maintained values: plain, "
                "refusal_prefix, refusal_tokens, harmful_prefix."
            )
        },
    )
    prefill_prefix: str = field(
        default="I cannot fulfill",
        metadata={"help": "assistant prefix used by the refusal_prefix condition"},
    )
    num_prefix_tokens: int = field(
        default=10,
        metadata={"help": "token prefix length used by refusal_tokens and harmful_prefix"},
    )
    max_samples: int = field(
        default=64,
        metadata={"help": "maximum number of HEx-PHI samples to probe; <=0 uses all samples"},
    )
    batch_size_per_device: int = field(default=4, metadata={"help": "the batch size"})
    max_length: int = field(
        default=2048,
        metadata={"help": "tokenizer truncation length for probing prompts; <=0 disables truncation"},
    )
    probe_positions: str = field(
        default="final_token",
        metadata={
            "help": (
                "Comma-separated token positions to probe. Maintained values: final_token, "
                "last_user_token, assistant_header_token, first_assistant_token."
            )
        },
    )
    save_dir: str = field(
        default="logs/safety_vector_probe",
        metadata={"help": "directory where vector probe artifacts are saved"},
    )
    vector_dtype: str = field(
        default="float16",
        metadata={"help": "dtype used when saving vectors: float16, bfloat16, or float32"},
    )
    save_vectors: bool = field(default=True, metadata={"help": "save raw layerwise vectors to NPZ"})
    use_cache: bool = field(default=False, metadata={"help": "use cache during forward passes"})


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str], token_positions: list[int]):
        self.prompts = prompts
        self.token_positions = token_positions

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> tuple[int, str, int]:
        return idx, self.prompts[idx], self.token_positions[idx]


def parse_conditions(raw_conditions: str) -> list[str]:
    conditions = [item.strip() for item in raw_conditions.split(",") if item.strip()]
    maintained_conditions = {"plain", "refusal_prefix", "refusal_tokens", "harmful_prefix"}
    unknown_conditions = sorted(set(conditions) - maintained_conditions)
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")
    if "plain" not in conditions:
        raise ValueError("The plain condition is required so directions can be computed against it")
    return conditions


def parse_probe_positions(raw_probe_positions: str) -> list[str]:
    probe_positions = [item.strip() for item in raw_probe_positions.split(",") if item.strip()]
    maintained_positions = {
        "final_token",
        "last_user_token",
        "assistant_header_token",
        "first_assistant_token",
    }
    unknown_positions = sorted(set(probe_positions) - maintained_positions)
    if unknown_positions:
        raise ValueError(f"Unknown probe_positions: {unknown_positions}")
    if not probe_positions:
        raise ValueError("At least one probe position is required")
    return probe_positions


def get_condition_messages(
    condition: str,
    tokenizer,
    prompt_style: str,
    system_prompt: str | None,
    input_template: str | None,
    output_header: str | None,
    prefill_prefix: str,
    num_prefix_tokens: int,
) -> tuple[list[list[dict[str, Any]]], list[Any]]:
    if condition == "plain":
        return get_eval_data.get_hex_phi(system_prompt, input_template, output_header)
    if condition == "refusal_prefix":
        return get_eval_data.get_hex_phi(system_prompt, input_template, prefill_prefix)
    if condition == "refusal_tokens":
        return get_eval_data.get_hex_phi_with_refusal_prefix(
            tokenizer,
            system_prompt,
            input_template,
            prompt_style,
            num_prefix_tokens,
        )
    if condition == "harmful_prefix":
        return get_eval_data.get_hex_phi_with_harmful_prefix(
            tokenizer,
            system_prompt,
            input_template,
            prompt_style,
            num_prefix_tokens,
        )
    raise ValueError(f"Condition {condition} not maintained")


def render_prompts(messages: list[list[dict[str, Any]]], generator: chat.Chat) -> list[str]:
    prompts = []
    for item in messages:
        conversation = generator.validate_conversation(item)
        prompts.append(generator.string_formatter({"messages": conversation})["text"])
    return prompts


def tokenization_kwargs_for_positions(args: ScriptArguments) -> dict[str, Any]:
    tokenization_kwargs = {
        "padding": False,
        "return_offsets_mapping": True,
    }
    if args.max_length > 0:
        tokenization_kwargs["truncation"] = True
        tokenization_kwargs["max_length"] = args.max_length
    return tokenization_kwargs


def tokenization_kwargs_for_position_lengths(args: ScriptArguments) -> dict[str, Any]:
    tokenization_kwargs = {"padding": False}
    if args.max_length > 0:
        tokenization_kwargs["truncation"] = True
        tokenization_kwargs["max_length"] = args.max_length
    return tokenization_kwargs


def last_real_token_index(offsets: list[tuple[int, int]]) -> int:
    for idx in range(len(offsets) - 1, -1, -1):
        start, end = offsets[idx]
        if end > start:
            return idx
    return max(0, len(offsets) - 1)


def token_index_overlapping_span(
    offsets: list[tuple[int, int]],
    start_char: int,
    end_char: int,
    *,
    first: bool,
) -> int | None:
    indices = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > start and start < end_char and end > start_char
    ]
    if not indices:
        return None
    return indices[0] if first else indices[-1]


def token_index_before_char(offsets: list[tuple[int, int]], char_position: int) -> int:
    for idx in range(len(offsets) - 1, -1, -1):
        start, end = offsets[idx]
        if end > start and end <= char_position:
            return idx
    return last_real_token_index(offsets)


def find_last_user_content_span(prompt: str, messages: list[dict[str, Any]]) -> tuple[int, int]:
    user_messages = [message for message in messages if message["role"] == "user"]
    if not user_messages:
        raise ValueError("Cannot locate last user token because the prompt has no user message")

    user_content = user_messages[-1]["content"]
    start_char = prompt.rfind(user_content)
    if start_char < 0:
        raise ValueError("Cannot locate user content in rendered prompt")
    return start_char, start_char + len(user_content)


def find_assistant_content_start(prompt: str, messages: list[dict[str, Any]]) -> int:
    assistant_content = messages[-1]["content"]
    if assistant_content == "":
        return len(prompt)

    start_char = prompt.rfind(assistant_content)
    if start_char < 0:
        raise ValueError("Cannot locate assistant prefix in rendered prompt")
    return start_char


def select_probe_token_index(
    prompt: str,
    messages: list[dict[str, Any]],
    offsets: list[tuple[int, int]],
    tokenizer,
    args: ScriptArguments,
    probe_position: str,
) -> int:
    if probe_position == "final_token":
        return last_real_token_index(offsets)

    if probe_position == "last_user_token":
        start_char, end_char = find_last_user_content_span(prompt, messages)
        token_index = token_index_overlapping_span(offsets, start_char, end_char, first=False)
        if token_index is None:
            raise ValueError("Cannot map last user content to a token index")
        return token_index

    assistant_start = find_assistant_content_start(prompt, messages)
    prompt_before_assistant_content = prompt[:assistant_start]
    prefix_token_count = len(
        tokenizer(
            prompt_before_assistant_content,
            **tokenization_kwargs_for_position_lengths(args),
        )["input_ids"]
    )
    if probe_position == "assistant_header_token":
        return max(0, prefix_token_count - 1)

    if probe_position == "first_assistant_token":
        assistant_content = messages[-1]["content"]
        if assistant_content == "":
            return max(0, prefix_token_count - 1)
        return min(prefix_token_count, len(offsets) - 1)

    raise ValueError(f"Probe position {probe_position} not maintained")


def compute_probe_token_positions(
    prompts: list[str],
    messages: list[list[dict[str, Any]]],
    tokenizer,
    args: ScriptArguments,
    probe_position: str,
) -> list[int]:
    token_positions = []
    tokenization_kwargs = tokenization_kwargs_for_positions(args)
    for prompt, sample_messages in zip(prompts, messages):
        encoded = tokenizer(prompt, **tokenization_kwargs)
        offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
        token_positions.append(
            select_probe_token_index(
                prompt=prompt,
                messages=sample_messages,
                offsets=offsets,
                tokenizer=tokenizer,
                args=args,
                probe_position=probe_position,
            )
        )
    return token_positions


def extract_sample_metadata(
    sample_idx: int,
    messages: list[dict[str, Any]],
    plain_text: Any | None,
) -> dict[str, Any]:
    user_messages = [message for message in messages if message["role"] == "user"]
    user_message = user_messages[0] if user_messages else {}

    category = user_message.get("category")
    if category is None and isinstance(plain_text, (list, tuple)) and len(plain_text) > 1:
        category = plain_text[1]

    return {
        "sample_index": sample_idx,
        "category": category,
        "user_prompt": user_message.get("content", ""),
    }


def selected_token_vectors(
    outputs,
    attention_mask: torch.Tensor,
    token_positions: torch.Tensor,
    padding_side: str,
) -> torch.Tensor:
    if padding_side == "left":
        sequence_lengths = attention_mask.long().sum(dim=1)
        selected_positions = token_positions.to(attention_mask.device) + (
            attention_mask.shape[1] - sequence_lengths
        )
    else:
        selected_positions = token_positions.to(attention_mask.device)

    batch_positions = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    return torch.stack(
        [hidden_state[batch_positions, selected_positions] for hidden_state in outputs.hidden_states],
        dim=1,
    )


def numpy_dtype_for_vectors(vector_dtype: str) -> np.dtype:
    if vector_dtype == "float16":
        return np.float16
    if vector_dtype == "bfloat16":
        # NumPy has no stable bfloat16 file dtype, so store bfloat16 requests as float32.
        return np.float32
    if vector_dtype == "float32":
        return np.float32
    raise ValueError("vector_dtype must be one of: float16, bfloat16, float32")


def collect_condition_vectors(
    model,
    tokenizer,
    prompts: list[str],
    token_positions: list[int],
    args: ScriptArguments,
    accelerator: Accelerator,
    condition_name: str,
    probe_position: str,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = PromptDataset(prompts, token_positions)
    data_loader = accelerator.prepare(
        DataLoader(dataset, batch_size=args.batch_size_per_device, shuffle=False)
    )

    model_for_forward = getattr(model, "module", model)
    vector_store: dict[int, np.ndarray] = {}
    length_store: dict[int, int] = {}
    save_dtype = numpy_dtype_for_vectors(args.vector_dtype)

    for batch_indices, batch_prompts, batch_token_positions in tqdm(
        data_loader,
        disable=not accelerator.is_local_main_process,
        desc=f"Probing {condition_name}/{probe_position}",
    ):
        tokenization_kwargs = {
            "padding": True,
            "return_tensors": "pt",
        }
        if args.max_length > 0:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = args.max_length

        model_inputs = tokenizer(list(batch_prompts), **tokenization_kwargs).to(accelerator.device)

        with torch.inference_mode():
            outputs = model_for_forward(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                output_hidden_states=True,
                use_cache=args.use_cache,
            )
            vectors = selected_token_vectors(
                outputs,
                model_inputs["attention_mask"],
                batch_token_positions,
                tokenizer.padding_side,
            )

        gathered_indices, gathered_lengths, gathered_positions, gathered_vectors = accelerator.gather_for_metrics(
            (
                batch_indices.to(accelerator.device),
                model_inputs["attention_mask"].sum(dim=1),
                batch_token_positions.to(accelerator.device),
                vectors,
            )
        )

        if accelerator.is_local_main_process:
            gathered_indices = gathered_indices.cpu().numpy()
            gathered_lengths = gathered_lengths.cpu().numpy()
            gathered_positions = gathered_positions.cpu().numpy()
            gathered_vectors = gathered_vectors.float().cpu().numpy().astype(save_dtype)
            for idx, length, position, vector in zip(
                gathered_indices,
                gathered_lengths,
                gathered_positions,
                gathered_vectors,
            ):
                vector_store[int(idx)] = vector
                length_store[int(idx)] = int(length)

        accelerator.wait_for_everyone()

    if not accelerator.is_local_main_process:
        return np.empty(0), np.empty(0)

    missing = sorted(set(range(len(prompts))) - set(vector_store))
    if missing:
        raise RuntimeError(f"Missing vectors for sample indices: {missing[:10]}")

    vectors_array = np.stack([vector_store[idx] for idx in range(len(prompts))], axis=0)
    lengths_array = np.array([length_store[idx] for idx in range(len(prompts))], dtype=np.int64)
    return vectors_array, lengths_array


def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    numerator = (a * b).sum(axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.maximum(denominator, eps)


def compute_layer_metrics(
    vectors_by_condition: dict[str, np.ndarray],
    conditions: list[str],
) -> list[dict[str, Any]]:
    plain = vectors_by_condition["plain"].astype(np.float32)
    num_samples, num_layers, _ = plain.shape
    train_count = max(1, num_samples // 2)
    eval_slice = slice(train_count, None) if num_samples - train_count > 0 else slice(0, num_samples)

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        if condition == "plain":
            continue
        compared = vectors_by_condition[condition].astype(np.float32)
        delta = compared - plain

        for layer_idx in range(num_layers):
            train_delta = delta[:train_count, layer_idx, :]
            eval_delta = delta[eval_slice, layer_idx, :]
            direction = train_delta.mean(axis=0)
            direction_norm = float(np.linalg.norm(direction))
            unit_direction = direction / max(direction_norm, 1e-8)

            projections = eval_delta @ unit_direction
            cosines = safe_cosine(eval_delta, unit_direction[None, :])
            delta_norms = np.linalg.norm(eval_delta, axis=-1)

            rows.append(
                {
                    "comparison": f"{condition}-plain",
                    "layer": layer_idx,
                    "n_train": int(train_delta.shape[0]),
                    "n_eval": int(eval_delta.shape[0]),
                    "direction_norm": direction_norm,
                    "delta_norm_mean": float(delta_norms.mean()),
                    "delta_norm_std": float(delta_norms.std()),
                    "projection_mean": float(projections.mean()),
                    "projection_std": float(projections.std()),
                    "cosine_mean": float(cosines.mean()),
                    "cosine_std": float(cosines.std()),
                }
            )

    return rows


def compute_direction_arrays(
    vectors_by_condition: dict[str, np.ndarray],
    conditions: list[str],
) -> tuple[list[str], np.ndarray]:
    plain = vectors_by_condition["plain"].astype(np.float32)
    num_samples = plain.shape[0]
    train_count = max(1, num_samples // 2)

    comparisons = []
    directions = []
    for condition in conditions:
        if condition == "plain":
            continue
        compared = vectors_by_condition[condition].astype(np.float32)
        comparisons.append(f"{condition}-plain")
        directions.append((compared[:train_count] - plain[:train_count]).mean(axis=0))

    if not directions:
        return comparisons, np.empty((0,) + plain.shape[1:], dtype=np.float32)
    return comparisons, np.stack(directions, axis=0)


def save_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()
    conditions = parse_conditions(args.conditions)
    probe_positions = parse_probe_positions(args.probe_positions)

    if args.eval_template not in common_eval_template:
        raise ValueError(f"eval_template {args.eval_template} not maintained")

    accelerator = Accelerator()
    save_dir = Path(args.save_dir)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    if accelerator.is_local_main_process:
        print(f"torch_dtype: {torch_dtype}")

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
    model = accelerator.prepare(model)

    eval_template = common_eval_template[args.eval_template]
    system_prompt = eval_template["system_prompt"]
    input_template = eval_template["input_template"]
    output_header = eval_template["output_header"]

    generator = chat.Chat(
        model=model,
        prompt_style=args.prompt_style,
        tokenizer=tokenizer,
        init_system_prompt=system_prompt,
    )

    messages_by_condition = {}
    plain_text_by_condition = {}
    with PartialState().local_main_process_first():
        for condition in conditions:
            messages, plain_text = get_condition_messages(
                condition=condition,
                tokenizer=tokenizer,
                prompt_style=args.prompt_style,
                system_prompt=system_prompt,
                input_template=input_template,
                output_header=output_header,
                prefill_prefix=args.prefill_prefix,
                num_prefix_tokens=args.num_prefix_tokens,
            )
            messages_by_condition[condition] = messages
            plain_text_by_condition[condition] = plain_text

    min_samples = min(len(messages_by_condition[condition]) for condition in conditions)
    if args.max_samples > 0:
        min_samples = min(min_samples, args.max_samples)
    if min_samples <= 0:
        raise ValueError("No samples available to probe")

    messages_by_condition = {
        condition: messages[:min_samples] for condition, messages in messages_by_condition.items()
    }
    plain_text_by_condition = {
        condition: plain_text[:min_samples] for condition, plain_text in plain_text_by_condition.items()
    }

    prompts_by_condition = {
        condition: render_prompts(messages, generator)
        for condition, messages in messages_by_condition.items()
    }

    token_positions_by_probe_and_condition = {
        probe_position: {
            condition: compute_probe_token_positions(
                prompts=prompts_by_condition[condition],
                messages=messages_by_condition[condition],
                tokenizer=tokenizer,
                args=args,
                probe_position=probe_position,
            )
            for condition in conditions
        }
        for probe_position in probe_positions
    }

    vectors_by_probe_and_condition = {}
    input_lengths_by_probe_and_condition = {}
    for probe_position in probe_positions:
        vectors_by_condition = {}
        input_lengths_by_condition = {}
        for condition in conditions:
            vectors, input_lengths = collect_condition_vectors(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts_by_condition[condition],
                token_positions=token_positions_by_probe_and_condition[probe_position][condition],
                args=args,
                accelerator=accelerator,
                condition_name=condition,
                probe_position=probe_position,
            )
            if accelerator.is_local_main_process:
                vectors_by_condition[condition] = vectors
                input_lengths_by_condition[condition] = input_lengths

        if accelerator.is_local_main_process:
            vectors_by_probe_and_condition[probe_position] = vectors_by_condition
            input_lengths_by_probe_and_condition[probe_position] = input_lengths_by_condition

    if not accelerator.is_local_main_process:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    direction_comparisons = None
    directions_by_probe = []
    for probe_position in probe_positions:
        vectors_by_condition = vectors_by_probe_and_condition[probe_position]
        position_metrics_rows = compute_layer_metrics(vectors_by_condition, conditions)
        for row in position_metrics_rows:
            row["probe_position"] = probe_position
        metrics_rows.extend(position_metrics_rows)

        position_direction_comparisons, position_directions = compute_direction_arrays(
            vectors_by_condition,
            conditions,
        )
        if direction_comparisons is None:
            direction_comparisons = position_direction_comparisons
        elif direction_comparisons != position_direction_comparisons:
            raise RuntimeError("Direction comparisons differ across probe positions")
        directions_by_probe.append(position_directions)

    if direction_comparisons is None:
        direction_comparisons = []
    directions = np.stack(directions_by_probe, axis=0)
    first_vectors_by_condition = vectors_by_probe_and_condition[probe_positions[0]]

    metrics_csv_path = save_dir / "layer_metrics.csv"
    save_metrics_csv(metrics_rows, metrics_csv_path)

    directions_path = save_dir / "directions.npz"
    np.savez_compressed(
        directions_path,
        probe_positions=np.array(probe_positions),
        comparisons=np.array(direction_comparisons),
        layer_indices=np.arange(first_vectors_by_condition["plain"].shape[1], dtype=np.int64),
        directions=directions,
    )

    vectors_path = None
    if args.save_vectors:
        vectors_path = save_dir / "boundary_vectors.npz"
        np.savez_compressed(
            vectors_path,
            probe_positions=np.array(probe_positions),
            condition_names=np.array(conditions),
            sample_indices=np.arange(min_samples, dtype=np.int64),
            layer_indices=np.arange(first_vectors_by_condition["plain"].shape[1], dtype=np.int64),
            vectors=np.stack(
                [
                    np.stack(
                        [
                            vectors_by_probe_and_condition[probe_position][condition]
                            for condition in conditions
                        ],
                        axis=0,
                    )
                    for probe_position in probe_positions
                ],
                axis=0,
            ),
            input_lengths=np.stack(
                [
                    np.stack(
                        [
                            input_lengths_by_probe_and_condition[probe_position][condition]
                            for condition in conditions
                        ],
                        axis=0,
                    )
                    for probe_position in probe_positions
                ],
                axis=0,
            ),
            token_positions=np.stack(
                [
                    np.stack(
                        [
                            np.array(
                                token_positions_by_probe_and_condition[probe_position][condition],
                                dtype=np.int64,
                            )
                            for condition in conditions
                        ],
                        axis=0,
                    )
                    for probe_position in probe_positions
                ],
                axis=0,
            ),
        )

    sample_metadata = [
        extract_sample_metadata(
            sample_idx=idx,
            messages=messages_by_condition["plain"][idx],
            plain_text=plain_text_by_condition["plain"][idx],
        )
        for idx in range(min_samples)
    ]

    metadata = {
        "args": asdict(args),
        "model_name_or_path": model_config.model_name_or_path,
        "conditions": conditions,
        "probe_positions": probe_positions,
        "num_samples": min_samples,
        "num_layers_including_embeddings": int(first_vectors_by_condition["plain"].shape[1]),
        "hidden_size": int(first_vectors_by_condition["plain"].shape[2]),
        "vectors_path": str(vectors_path) if vectors_path is not None else None,
        "directions_path": str(directions_path),
        "metrics_csv_path": str(metrics_csv_path),
        "samples": sample_metadata,
    }
    metadata_path = save_dir / "metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    top_rows = sorted(metrics_rows, key=lambda row: abs(row["projection_mean"]), reverse=True)[:8]
    print(f"Saved metadata to {metadata_path}")
    if vectors_path is not None:
        print(f"Saved vectors to {vectors_path}")
    print(f"Saved directions to {directions_path}")
    print(f"Saved layer metrics to {metrics_csv_path}")
    print("Top projection layers:")
    for row in top_rows:
        print(
            f"  {row['probe_position']} {row['comparison']} layer={row['layer']} "
            f"projection_mean={row['projection_mean']:.4f} "
            f"cosine_mean={row['cosine_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
