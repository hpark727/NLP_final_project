from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import random
from typing import Any

from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from finetuning_buckets.inference import chat
from finetuning_buckets.inference.safety_eval.evaluator import common_eval_template
from finetuning_buckets.models import get_model


@dataclass
class ScriptArguments:
    model_family: str = field(default="llama3", metadata={"help": "the model family"})
    prompt_style: str = field(default="llama3", metadata={"help": "the string prompt style"})
    eval_template: str = field(default="plain", metadata={"help": "the eval template"})
    max_samples_per_class: int = field(
        default=256,
        metadata={"help": "maximum harmful and benign samples to use per class; <=0 uses all"},
    )
    train_fraction: float = field(default=0.7, metadata={"help": "fraction used for probe train"})
    batch_size_per_device: int = field(default=4, metadata={"help": "feature extraction batch size"})
    max_length: int = field(default=2048, metadata={"help": "tokenizer truncation length"})
    seed: int = field(default=0, metadata={"help": "random seed for sampling and split"})
    probe_steps: int = field(default=600, metadata={"help": "AdamW optimization steps per layer"})
    probe_lr: float = field(default=0.05, metadata={"help": "probe optimizer learning rate"})
    l2_weight: float = field(default=0.01, metadata={"help": "L2 penalty on probe weights"})
    save_dir: str = field(
        default="logs/harmfulness_linear_probe",
        metadata={"help": "directory where probe artifacts are saved"},
    )
    use_cache: bool = field(default=False, metadata={"help": "use cache during forward passes"})


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str], token_positions: list[int], labels: list[int]):
        self.prompts = prompts
        self.token_positions = token_positions
        self.labels = labels

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> tuple[int, str, int, int]:
        return idx, self.prompts[idx], self.token_positions[idx], self.labels[idx]


def load_harmful_prompts() -> list[dict[str, Any]]:
    prompts = []
    with open("finetuning_buckets/datasets/data/safety_bench/HEx-PHI.jsonl") as handle:
        for line in handle:
            item = json.loads(line)
            prompts.append(
                {
                    "text": item["instruction"],
                    "category": item.get("category"),
                    "source": "hex_phi",
                    "label": 1,
                }
            )
    return prompts


def load_benign_prompts() -> list[dict[str, Any]]:
    with open("finetuning_buckets/datasets/data/tasks/data_augmentation/llama2_alpaca_anchor.json") as handle:
        data = json.load(handle)

    prompts = []
    for item in data:
        instruction = item["instruction"]
        if item.get("input", "").strip():
            instruction = f"{instruction}\n\n{item['input']}"
        prompts.append(
            {
                "text": instruction,
                "category": "benign",
                "source": "alpaca_anchor",
                "label": 0,
            }
        )
    return prompts


def sample_balanced_examples(args: ScriptArguments) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    harmful = load_harmful_prompts()
    benign = load_benign_prompts()
    rng.shuffle(harmful)
    rng.shuffle(benign)

    if args.max_samples_per_class > 0:
        harmful = harmful[: args.max_samples_per_class]
        benign = benign[: args.max_samples_per_class]
    else:
        count = min(len(harmful), len(benign))
        harmful = harmful[:count]
        benign = benign[:count]

    examples = harmful + benign
    rng.shuffle(examples)
    return examples


def stratified_split(labels: list[int], train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train_indices = []
    eval_indices = []
    for label in sorted(set(labels)):
        indices = [idx for idx, item_label in enumerate(labels) if item_label == label]
        rng.shuffle(indices)
        train_count = max(1, min(len(indices) - 1, int(round(len(indices) * train_fraction))))
        train_indices.extend(indices[:train_count])
        eval_indices.extend(indices[train_count:])

    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)
    return train_indices, eval_indices


def render_prompt(example: dict[str, Any], generator: chat.Chat) -> str:
    messages = [
        {"role": "user", "content": example["text"]},
        {"role": "assistant", "content": ""},
    ]
    conversation = generator.validate_conversation(messages)
    return generator.string_formatter({"messages": conversation})["text"]


def tokenization_kwargs(args: ScriptArguments, *, offsets: bool = False) -> dict[str, Any]:
    kwargs = {"padding": False}
    if offsets:
        kwargs["return_offsets_mapping"] = True
    if args.max_length > 0:
        kwargs["truncation"] = True
        kwargs["max_length"] = args.max_length
    return kwargs


def find_last_user_token(prompt: str, user_text: str, tokenizer, args: ScriptArguments) -> int:
    encoded = tokenizer(prompt, **tokenization_kwargs(args, offsets=True))
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    user_start = prompt.rfind(user_text)
    if user_start < 0:
        raise ValueError("Cannot locate user text in rendered prompt")
    user_end = user_start + len(user_text)

    user_token_indices = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > start and start < user_end and end > user_start
    ]
    if not user_token_indices:
        raise ValueError("Cannot map user text to token offsets")
    return user_token_indices[-1]


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


def collect_features(
    model,
    tokenizer,
    prompts: list[str],
    token_positions: list[int],
    labels: list[int],
    args: ScriptArguments,
    accelerator: Accelerator,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = PromptDataset(prompts, token_positions, labels)
    data_loader = accelerator.prepare(
        DataLoader(dataset, batch_size=args.batch_size_per_device, shuffle=False)
    )

    model_for_forward = getattr(model, "module", model)
    feature_store: dict[int, np.ndarray] = {}
    label_store: dict[int, int] = {}

    for batch_indices, batch_prompts, batch_token_positions, batch_labels in tqdm(
        data_loader,
        disable=not accelerator.is_local_main_process,
        desc="Extracting hidden states",
    ):
        model_inputs = tokenizer(
            list(batch_prompts),
            padding=True,
            return_tensors="pt",
            truncation=args.max_length > 0,
            max_length=args.max_length if args.max_length > 0 else None,
        ).to(accelerator.device)

        with torch.inference_mode():
            outputs = model_for_forward(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                output_hidden_states=True,
                use_cache=args.use_cache,
            )
            features = selected_token_vectors(
                outputs,
                model_inputs["attention_mask"],
                batch_token_positions,
                tokenizer.padding_side,
            )

        gathered_indices, gathered_labels, gathered_features = accelerator.gather_for_metrics(
            (
                batch_indices.to(accelerator.device),
                batch_labels.to(accelerator.device),
                features,
            )
        )

        if accelerator.is_local_main_process:
            gathered_indices = gathered_indices.cpu().numpy()
            gathered_labels = gathered_labels.cpu().numpy()
            gathered_features = gathered_features.float().cpu().numpy().astype(np.float32)
            for idx, label, feature in zip(gathered_indices, gathered_labels, gathered_features):
                feature_store[int(idx)] = feature
                label_store[int(idx)] = int(label)

        accelerator.wait_for_everyone()

    if not accelerator.is_local_main_process:
        return np.empty(0), np.empty(0)

    missing = sorted(set(range(len(prompts))) - set(feature_store))
    if missing:
        raise RuntimeError(f"Missing features for sample indices: {missing[:10]}")

    features = np.stack([feature_store[idx] for idx in range(len(prompts))], axis=0)
    labels = np.array([label_store[idx] for idx in range(len(prompts))], dtype=np.int64)
    return features, labels


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / (np.arange(len(labels)) + 1)
    return float((precision * sorted_labels).sum() / positive_count)


def train_layer_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    args: ScriptArguments,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    x_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    x_eval = torch.tensor(eval_features, dtype=torch.float32)

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_eval = (x_eval - mean) / std

    linear = torch.nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=args.probe_lr, weight_decay=0.0)

    for _ in range(args.probe_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = linear(x_train).squeeze(-1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        l2 = linear.weight.square().sum()
        loss = bce + args.l2_weight * l2 / x_train.shape[0]
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        train_scores = torch.sigmoid(linear(x_train).squeeze(-1)).numpy()
        eval_scores = torch.sigmoid(linear(x_eval).squeeze(-1)).numpy()

    eval_predictions = (eval_scores >= 0.5).astype(np.int64)
    accuracy = float((eval_predictions == eval_labels).mean())
    positive_mask = eval_labels == 1
    negative_mask = eval_labels == 0
    true_positive_rate = float((eval_predictions[positive_mask] == 1).mean()) if positive_mask.any() else float("nan")
    true_negative_rate = float((eval_predictions[negative_mask] == 0).mean()) if negative_mask.any() else float("nan")

    return {
        "train_auroc": binary_auc(train_labels, train_scores),
        "accuracy": accuracy,
        "balanced_accuracy": float((true_positive_rate + true_negative_rate) / 2),
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "auroc": binary_auc(eval_labels, eval_scores),
        "auprc": average_precision(eval_labels, eval_scores),
    }


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()
    if args.eval_template not in common_eval_template:
        raise ValueError(f"eval_template {args.eval_template} not maintained")

    accelerator = Accelerator()
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
    generator = chat.Chat(
        model=model,
        prompt_style=args.prompt_style,
        tokenizer=tokenizer,
        init_system_prompt=eval_template["system_prompt"],
    )

    examples = sample_balanced_examples(args)
    prompts = [render_prompt(example, generator) for example in examples]
    token_positions = [
        find_last_user_token(prompt, example["text"], tokenizer, args)
        for prompt, example in zip(prompts, examples)
    ]
    labels = [example["label"] for example in examples]

    features, labels_array = collect_features(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        token_positions=token_positions,
        labels=labels,
        args=args,
        accelerator=accelerator,
    )
    if not accelerator.is_local_main_process:
        return

    train_indices, eval_indices = stratified_split(labels, args.train_fraction, args.seed)
    train_labels = labels_array[train_indices]
    eval_labels = labels_array[eval_indices]

    rows = []
    for layer_idx in tqdm(range(features.shape[1]), desc="Training layer probes"):
        metrics = train_layer_probe(
            train_features=features[train_indices, layer_idx, :],
            train_labels=train_labels,
            eval_features=features[eval_indices, layer_idx, :],
            eval_labels=eval_labels,
            args=args,
            seed=args.seed + layer_idx,
        )
        rows.append(
            {
                "layer": layer_idx,
                "position": "last_user_token",
                "n_train": int(len(train_indices)),
                "n_eval": int(len(eval_indices)),
                "n_train_harmful": int(train_labels.sum()),
                "n_train_benign": int(len(train_labels) - train_labels.sum()),
                "n_eval_harmful": int(eval_labels.sum()),
                "n_eval_benign": int(len(eval_labels) - eval_labels.sum()),
                **metrics,
            }
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "layer_metrics.csv"
    save_csv(rows, metrics_path)

    metadata = {
        "args": asdict(args),
        "model_name_or_path": model_config.model_name_or_path,
        "num_samples": len(examples),
        "num_harmful": int(sum(labels)),
        "num_benign": int(len(labels) - sum(labels)),
        "num_layers_including_embeddings": int(features.shape[1]),
        "hidden_size": int(features.shape[2]),
        "metrics_csv_path": str(metrics_path),
        "examples": examples,
    }
    metadata_path = save_dir / "metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    top_rows = sorted(rows, key=lambda row: row["auroc"], reverse=True)[:8]
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved layer metrics to {metrics_path}")
    print("Top AUROC layers:")
    for row in top_rows:
        print(
            f"  layer={row['layer']} auroc={row['auroc']:.4f} "
            f"auprc={row['auprc']:.4f} balanced_accuracy={row['balanced_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
