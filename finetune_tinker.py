from __future__ import annotations
import os
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
ENV_PATH = Path(".env")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name",
        required=True,
        help="Tinker base model name, e.g. meta-llama/Llama-3.2-3B",
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Weight on the safety-recovery objective term; the utility anchor gets weight (1 - alpha).",
    )
    parser.add_argument(
        "--anchor_batch_size",
        type=int,
        default=None,
        help="Optional manual override for the anchor batch size. If omitted, it is derived from alpha.",
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--log_path",
        default="logs/tinker_finetune",
        help="Directory for local checkpoint metadata written by Tinker checkpoint utils.",
    )
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
        from tinker_cookbook import checkpoint_utils, model_info, renderers
        from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
        from tinker_cookbook.supervised.data import conversation_to_datum
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:
        raise SystemExit(
            "Failed to import Tinker dependencies.\n"
            f"Underlying error: {type(exc).__name__}: {exc}\n\n"
            "If `tinker` / `tinker-cookbook` are missing, install them in your active venv.\n"
            "If they are already installed, a transitive dependency is probably missing.\n"
            "Common fix:\n"
            "  pip install orjson\n"
            "or reinstall:\n"
            "  pip install -U tinker tinker-cookbook"
        ) from exc

    return (
        tinker,
        checkpoint_utils,
        model_info,
        renderers,
        compute_mean_nll,
        datum_from_model_input_weights,
        conversation_to_datum,
        get_tokenizer,
    )

def load_dotenv_if_available() -> None:
    if not ENV_PATH.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        print(
            "Note: `.env` detected but `python-dotenv` is not installed. "
            "Install it with `uv pip install python-dotenv` to auto-load environment variables."
        )
        return

    load_dotenv(ENV_PATH, override=False)



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


def _normalize_nonzero_weights(weights):
    total = float(weights.sum())
    if total > 0:
        weights = weights / total
    return weights


def _scale_weights(weights, scale: float):
    if scale <= 0:
        return weights * 0.0
    return weights * scale


def compute_anchor_batch_size(safety_batch_size: int, alpha: float) -> int:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in the interval (0, 1].")
    if alpha == 1.0:
        return 0
    return max(1, round(safety_batch_size * (1.0 - alpha) / alpha))


def build_mean_reduction_datum(
    messages,
    renderer,
    max_length,
    datum_from_model_input_weights,
    weight_scale: float = 1.0,
):
    model_input, weights = renderer.build_supervised_example(messages)
    weights = _normalize_nonzero_weights(weights)
    weights = _scale_weights(weights, weight_scale)
    return datum_from_model_input_weights(
        model_input,
        weights,
        max_length=max_length,
    )


def build_augmented_datum(
    example: dict,
    renderer,
    tokenizer,
    max_length: int,
    max_harmful_prefix_tokens: int,
    rng: random.Random,
    datum_from_model_input_weights,
    weight_scale: float = 1.0,
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

    weights = _normalize_nonzero_weights(weights)
    weights = _scale_weights(weights, weight_scale)

    return datum_from_model_input_weights(
        model_input,
        weights,
        max_length=max_length,
    )


def main() -> None:
    args = parse_args()
    load_dotenv_if_available()
    (
        tinker,
        checkpoint_utils,
        model_info,
        renderers,
        compute_mean_nll,
        datum_from_model_input_weights,
        conversation_to_datum,
        get_tokenizer,
    ) = _require_tinker()
    
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY is not set. Put it in `.env` or export it in your shell before running."
        )


    rng = random.Random(args.seed)

    if not DATA_AUG_PATH.exists():
        raise SystemExit(f"Missing augmentation dataset: {DATA_AUG_PATH}")
    if (not args.disable_anchor) and (not ANCHOR_PATH.exists()):
        raise SystemExit(f"Missing anchor dataset: {ANCHOR_PATH}")
    if not (0.0 < args.alpha <= 1.0):
        raise SystemExit("--alpha must be in the interval (0, 1].")

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

    if args.disable_anchor:
        anchor_batch_size = 0
    elif args.anchor_batch_size is not None:
        anchor_batch_size = args.anchor_batch_size
    else:
        anchor_batch_size = compute_anchor_batch_size(args.batch_size, args.alpha)

    if anchor_batch_size == 0:
        safety_example_weight = 1.0
        anchor_example_weight = 0.0
        effective_alpha = 1.0
    else:
        safety_example_weight = args.alpha / args.batch_size
        anchor_example_weight = (1.0 - args.alpha) / anchor_batch_size
        effective_alpha = (
            args.batch_size * safety_example_weight
            / (
                args.batch_size * safety_example_weight
                + anchor_batch_size * anchor_example_weight
            )
        )

    print(f"Using renderer: {renderer_name}")
    print(f"Training rows: {len(train_rows)}")
    print(f"Anchor rows: {len(anchor_rows)}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Alpha: {args.alpha}")
    print(f"Anchor batch size: {anchor_batch_size}")
    print(f"Per-safety-example weight: {safety_example_weight}")
    print(f"Per-anchor-example weight: {anchor_example_weight}")
    print(f"Effective alpha this batching implements: {effective_alpha}")

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
                            weight_scale=safety_example_weight,
                        )
                    if datum is None:
                        datum = build_mean_reduction_datum(
                            build_refusal_messages(example),
                            renderer,
                            args.max_length,
                            datum_from_model_input_weights,
                            weight_scale=safety_example_weight,
                        )
                else:
                    datum = build_mean_reduction_datum(
                        build_refusal_messages(example),
                        renderer,
                        args.max_length,
                        datum_from_model_input_weights,
                        weight_scale=safety_example_weight,
                    )
                batch.append(datum)

            if anchor_rows:
                for _ in range(anchor_batch_size):
                    anchor_example = anchor_rows[rng.randrange(len(anchor_rows))]
                    batch.append(
                        build_mean_reduction_datum(
                            build_anchor_messages(anchor_example),
                            renderer,
                            args.max_length,
                            datum_from_model_input_weights,
                            weight_scale=anchor_example_weight,
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
    checkpoint_paths = checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=args.log_path,
        kind="both",
        loop_state={"step": global_step, "epoch": args.num_epochs},
        ttl_seconds=None,
    )
    print("Saved checkpoint paths:")
    print(json.dumps(checkpoint_paths, indent=2))


if __name__ == "__main__":
    main()
