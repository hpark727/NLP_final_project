"""
Train the paper's safety-augmentation method with Thinking Machines Tinker.

This script keeps the core idea of the original repo intact:
1. build refusal-only supervised examples, or
2. build augmented examples that prepend a harmful assistant prefix and only
   train on the refusal continuation.

Unlike the Hugging Face trainer path, this script uses Tinker's LoRA training
API directly.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


DATA_AUG_PATH = Path(
    "finetuning_buckets/datasets/data/tasks/data_augmentation/llama2_safety_data_direct.jsonl"
)
ANCHOR_PATH = Path(
    "finetuning_buckets/datasets/data/tasks/data_augmentation/llama2_alpaca_anchor.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name",
        required=True,
        help="Tinker base model name, e.g. meta-llama/Llama-3.2-3B-Instruct",
    )
    parser.add_argument(
        "--renderer_name",
        default=None,
        help="Optional Tinker renderer override. Defaults to the recommended renderer for the model.",
    )
    parser.add_argument("--base_url", default=None, help="Optional Tinker API base URL.")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--anchor_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--augment_probability",
        type=float,
        default=0.5,
        help="Probability of using harmful-prefix augmentation instead of refusal-only training.",
    )
    parser.add_argument(
        "--max_harmful_prefix_tokens",
        type=int,
        default=100,
        help="Maximum number of harmful assistant tokens to splice in before the refusal.",
    )
    parser.add_argument(
        "--disable_anchor",
        action="store_true",
        help="Disable the benign anchor batch used in the paper's augmentation setup.",
    )
    return parser.parse_args()


def _require_tinker():
    try:
        import tinker
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
        from tinker_cookbook.supervised.data import conversation_to_datum
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:
        raise SystemExit(
            "This script requires `tinker` and `tinker-cookbook`.\n"
            "Install them with:\n"
            "  uv pip install tinker tinker-cookbook\n"
            "and set TINKER_API_KEY before running."
        ) from exc

    return tinker, model_info, renderers, compute_mean_nll, datum_from_model_input_weights, conversation_to_datum, get_tokenizer


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)


def build_user_text(example: dict) -> str:
    if ("input" not in example) or (not str(example["input"]).strip()):
        return str(example["instruction"])
    return f"{example['instruction']}\n\n{example['input']}"


def build_refusal_messages(example: dict) -> list[dict]:
    return [
        {"role": "user", "content": build_user_text(example)},
        {"role": "assistant", "content": str(example["refusal"])},
    ]


def build_anchor_messages(example: dict) -> list[dict]:
    return [
        {"role": "user", "content": build_user_text(example)},
        {"role": "assistant", "content": str(example["output"])},
    ]


def build_augmented_datum(
    example: dict,
    renderer,
    tokenizer,
    max_length: int,
    max_harmful_prefix_tokens: int,
    rng: random.Random,
    datum_from_model_input_weights,
):
    harmful_tokens = tokenizer.encode(str(example["harmful"]), add_special_tokens=False)
    if not harmful_tokens:
        return None

    cutoff = rng.randint(1, min(len(harmful_tokens), max_harmful_prefix_tokens))
    harmful_prefix_tokens = harmful_tokens[:cutoff]
    harmful_prefix = tokenizer.decode(harmful_prefix_tokens)

    messages = [
        {"role": "user", "content": build_user_text(example)},
        {"role": "assistant", "content": harmful_prefix + str(example["refusal"])},
    ]

    model_input, weights = renderer.build_supervised_example(messages)
    positive_weight_positions = weights.nonzero().flatten()
    prefix_token_count = len(harmful_prefix_tokens)

    if prefix_token_count > 0 and len(positive_weight_positions) >= prefix_token_count:
        weights[positive_weight_positions[:prefix_token_count]] = 0.0

    return datum_from_model_input_weights(
        model_input,
        weights,
        max_length=max_length,
        reduction="mean",
    )


def main() -> None:
    args = parse_args()

    (
        tinker,
        model_info,
        renderers,
        compute_mean_nll,
        datum_from_model_input_weights,
        conversation_to_datum,
        get_tokenizer,
    ) = _require_tinker()

    rng = random.Random(args.seed)

    if not DATA_AUG_PATH.exists():
        raise SystemExit(f"Missing augmentation dataset: {DATA_AUG_PATH}")
    if (not args.disable_anchor) and (not ANCHOR_PATH.exists()):
        raise SystemExit(f"Missing anchor dataset: {ANCHOR_PATH}")

    train_rows = load_jsonl(DATA_AUG_PATH)
    anchor_rows = [] if args.disable_anchor else load_json(ANCHOR_PATH)

    tokenizer = get_tokenizer(args.model_name)
    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=args.model_name)

    service_client = tinker.ServiceClient(base_url=args.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=args.model_name,
        rank=args.lora_rank,
    )

    steps_per_epoch = len(train_rows) // args.batch_size
    if steps_per_epoch == 0:
        raise SystemExit("Batch size is larger than the number of augmentation rows.")

    print(f"Using renderer: {renderer_name}")
    print(f"Training rows: {len(train_rows)}")
    print(f"Anchor rows: {len(anchor_rows)}")
    print(f"Steps per epoch: {steps_per_epoch}")

    global_step = 0
    for epoch in range(args.num_epochs):
        epoch_rows = list(train_rows)
        rng.shuffle(epoch_rows)

        for batch_idx in range(steps_per_epoch):
            start_time = time.time()
            batch_slice = epoch_rows[
                batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
            ]

            batch = []
            for example in batch_slice:
                use_aug = rng.random() < args.augment_probability
                if use_aug:
                    datum = build_augmented_datum(
                        example,
                        renderer,
                        tokenizer,
                        args.max_length,
                        args.max_harmful_prefix_tokens,
                        rng,
                        datum_from_model_input_weights,
                    )
                    if datum is None:
                        datum = conversation_to_datum(
                            build_refusal_messages(example),
                            renderer,
                            args.max_length,
                            reduction="mean",
                        )
                else:
                    datum = conversation_to_datum(
                        build_refusal_messages(example),
                        renderer,
                        args.max_length,
                        reduction="mean",
                    )
                batch.append(datum)

            if anchor_rows:
                for _ in range(args.anchor_batch_size):
                    anchor_example = anchor_rows[rng.randrange(len(anchor_rows))]
                    batch.append(
                        conversation_to_datum(
                            build_anchor_messages(anchor_example),
                            renderer,
                            args.max_length,
                            reduction="mean",
                        )
                    )

            adam_params = tinker.AdamParams(
                learning_rate=args.learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            if (global_step % args.log_every) == 0:
                train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
                train_weights = [d.loss_fn_inputs["weights"] for d in batch]
                train_nll = compute_mean_nll(train_logprobs, train_weights)
                metrics = dict(optim_result.metrics or {})
                metrics.update(
                    step=global_step,
                    epoch=epoch,
                    train_mean_nll=train_nll,
                    num_sequences=len(batch),
                    num_tokens=sum(d.model_input.length for d in batch),
                    time_total=time.time() - start_time,
                )
                print(json.dumps(metrics))

            global_step += 1

    print("Training loop finished.")
    print(
        "Tip: save weights or publish the resulting checkpoint with Tinker's checkpoint utilities "
        "or the Tinker Console."
    )


if __name__ == "__main__":
    main()
