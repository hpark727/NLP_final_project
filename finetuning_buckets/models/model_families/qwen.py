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
        Path("ckpts/Qwen/Qwen3-4B-Instruct-2507"),
        Path("ckpts/Qwen3-4B-Instruct-2507"),
        adapter_path.parents[1] / "Qwen" / "Qwen3-4B-Instruct-2507",
        adapter_path.parents[0] / "Qwen" / "Qwen3-4B-Instruct-2507",
    ]
    for candidate_path in candidate_paths:
        if (candidate_path / "config.json").exists():
            return str(candidate_path)

    raise ValueError(
        "LoRA adapter_config.json does not specify base_model_name_or_path, "
        "and ckpts/Qwen/Qwen3-4B-Instruct-2507 was not found."
    )


def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    model_path = Path(model_name_or_path)
    is_adapter = model_path.is_dir() and (model_path / "adapter_config.json").exists()
    trust_remote_code = model_kwargs.get("trust_remote_code", False)

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

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        add_eos_token=False,
        add_bos_token=False,
    )

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = padding_side

    return model, tokenizer


class QwenStringConverter:
    @staticmethod
    def _normalize_messages(example):
        if "messages" not in example:
            raise ValueError("No messages in the example")

        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("No messages in the example")
        return messages

    @staticmethod
    def _append_qwen_message(parts, role, content, close=True):
        parts.append(f"<|im_start|>{role}\n{content}")
        if close:
            parts.append("<|im_end|>\n")

    @staticmethod
    def string_formatter(example):
        messages = QwenStringConverter._normalize_messages(example)

        parts = []
        pt = 0
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            pt = 1
        else:
            system_prompt = default_system_prompt

        if system_prompt:
            QwenStringConverter._append_qwen_message(parts, "system", system_prompt)

        if pt == len(messages):
            raise ValueError("the message should be user - assistant alternation")

        while pt < len(messages):
            if messages[pt]["role"] != "user":
                raise ValueError("the message should be user - assistant alternation")
            QwenStringConverter._append_qwen_message(parts, "user", messages[pt]["content"])
            pt += 1

            if pt >= len(messages) or messages[pt]["role"] != "assistant":
                raise ValueError("the message should be user - assistant alternation")
            QwenStringConverter._append_qwen_message(parts, "assistant", messages[pt]["content"])
            pt += 1

        return {"text": "".join(parts)}

    @staticmethod
    def string_formatter_completion_only(example):
        messages = QwenStringConverter._normalize_messages(example)
        if messages[-1]["role"] != "assistant":
            raise ValueError("completion only mode should end with a header of assistant message")

        parts = []
        pt = 0
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            pt = 1
        else:
            system_prompt = default_system_prompt

        if system_prompt:
            QwenStringConverter._append_qwen_message(parts, "system", system_prompt)

        while pt < len(messages) - 1:
            if messages[pt]["role"] != "user":
                raise ValueError("the message should be user - assistant alternation")
            QwenStringConverter._append_qwen_message(parts, "user", messages[pt]["content"])
            pt += 1

            if pt >= len(messages) - 1:
                break
            if messages[pt]["role"] != "assistant":
                raise ValueError("the message should be user - assistant alternation")
            QwenStringConverter._append_qwen_message(parts, "assistant", messages[pt]["content"])
            pt += 1

        QwenStringConverter._append_qwen_message(
            parts,
            "assistant",
            messages[-1]["content"],
            close=False,
        )
        return {"text": "".join(parts)}

    @staticmethod
    def conversion_to_qwen_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(QwenStringConverter.string_formatter, remove_columns=redundant_columns)
        return dataset


def qwen_response_template_ids(tokenizer):
    return [tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)]


class AugmentedSafetyDataCollator:
    def __new__(cls, tokenizer, *args, **kwargs):
        from finetuning_buckets.models.model_families.llama2 import AugmentedSafetyDataCollator

        kwargs.setdefault("response_template", qwen_response_template_ids(tokenizer))
        return AugmentedSafetyDataCollator(tokenizer=tokenizer, *args, **kwargs)
