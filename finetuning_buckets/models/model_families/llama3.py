import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


default_system_prompt = ""


def _resolve_adapter_base_model(adapter_path: Path) -> str:
    with (adapter_path / "adapter_config.json").open() as handle:
        adapter_config = json.load(handle)

    base_model_name_or_path = adapter_config.get("base_model_name_or_path")
    if base_model_name_or_path:
        return base_model_name_or_path

    candidate_paths = [
        Path("ckpts/meta-llama/Llama-3.2-3B"),
        adapter_path.parents[1] / "meta-llama" / "Llama-3.2-3B",
        adapter_path.parents[0] / "meta-llama" / "Llama-3.2-3B",
    ]
    for candidate_path in candidate_paths:
        if (candidate_path / "config.json").exists():
            return str(candidate_path)

    raise ValueError(
        "LoRA adapter_config.json does not specify base_model_name_or_path, "
        "and ckpts/meta-llama/Llama-3.2-3B was not found."
    )


def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    model_path = Path(model_name_or_path)
    is_adapter = model_path.is_dir() and (model_path / "adapter_config.json").exists()

    tokenizer_path = model_name_or_path
    if is_adapter:
        from peft import PeftModel

        base_model_name_or_path = _resolve_adapter_base_model(model_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, **model_kwargs)
        model = PeftModel.from_pretrained(model, model_name_or_path)
        tokenizer_path = base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        tokenizer_path = model.config._name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = padding_side

    return model, tokenizer
