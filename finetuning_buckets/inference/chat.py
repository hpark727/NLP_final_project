import torch


class Chat:
    # by default, will maintain all conversations in OpenAI chat format

    def __init__(self, model, prompt_style, tokenizer, max_length=2048,
                 init_conversation=None, init_system_prompt=None):

        if init_conversation is not None and init_system_prompt is not None:
            raise ValueError("init_conversation and init_system_prompt cannot be provided at the same time")

        self.model = model
        self.prompt_style = prompt_style
        self.tokenizer = tokenizer
        self.max_length = max_length  # limit the length of the whole conversation

        # formatter will be used to convert openai chat format to string
        if prompt_style == 'llama2':
            from finetuning_buckets.models.model_families.llama2 import LlamaStringConverter, default_system_prompt
            self.string_formatter = LlamaStringConverter.string_formatter_completion_only
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = None
        elif prompt_style == 'gemma':
            from finetuning_buckets.models.model_families.gemma import GemmaStringConverter, default_system_prompt
            self.string_formatter = GemmaStringConverter.string_formatter_completion_only
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = None
        elif prompt_style == 'llama2_base':
            from finetuning_buckets.models.model_families.llama2_base import LlamaStringConverter, default_system_prompt, base_stopping_criteria
            self.string_formatter = LlamaStringConverter.string_formatter_completion_only
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = base_stopping_criteria
        elif prompt_style == 'gemma_base':
            from finetuning_buckets.models.model_families.gemma_base import GemmaStringConverter, default_system_prompt, base_stopping_criteria
            self.string_formatter = GemmaStringConverter.string_formatter_completion_only
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = base_stopping_criteria
        elif prompt_style == 'llama3':
            from finetuning_buckets.models.model_families.llama3 import default_system_prompt
            if getattr(tokenizer, "chat_template", None):
                self.string_formatter = self._string_formatter_completion_only_llama3
            else:
                self.string_formatter = self._string_formatter_completion_only_llama3_base
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = None
        else:
            raise ValueError(f"Prompt style {prompt_style} not supported")

        if init_conversation is not None:
            self.validate_conversation(init_conversation)
            if isinstance(init_conversation, dict):
                init_conversation = init_conversation['messages']

            if init_conversation[-1]['role'] == 'user':
                raise ValueError("the last message of init_conversation should be assistant message or system prompt, not user message")

            if init_conversation[0]['role'] != 'system':
                self.system_prompt = self.default_system_prompt
                self.conversation = self.init_conversation() + init_conversation
            else:
                self.system_prompt = init_conversation[0]['content']
                self.conversation = init_conversation
        else:
            if init_system_prompt is not None:
                self.system_prompt = init_system_prompt
            else:
                self.system_prompt = self.default_system_prompt

            self.conversation = self.init_conversation()

    def _get_model_for_generation(self):
        return getattr(self.model, "module", self.model)

    def _get_model_device(self):
        model_for_generation = self._get_model_for_generation()
        return next(model_for_generation.parameters()).device

    def _validate_distribution_tracking_steps(self, k):
        if k is None:
            return None
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer when tracking token distributions")
        return k

    def _extract_tracked_token_distributions(self, generation_outputs, input_width, k):
        tracked_steps = min(k, len(generation_outputs.scores))
        step_scores = torch.stack(generation_outputs.scores[:tracked_steps], dim=1)
        token_distributions = torch.softmax(step_scores, dim=-1).detach().cpu()
        generated_token_ids = generation_outputs.sequences[:, input_width : input_width + tracked_steps].detach().cpu()

        return {
            "tracked_steps": tracked_steps,
            "generated_token_ids": generated_token_ids,
            "token_distributions": token_distributions,
        }

    def _string_formatter_completion_only_llama3(self, example):
        if 'messages' not in example:
            raise ValueError("No messages in the example")

        messages = example['messages']
        if len(messages) == 0:
            raise ValueError("No messages in the example")
        if messages[-1]['role'] != 'assistant':
            raise ValueError("completion only mode should end with a header of assistant message")

        prompt_messages = messages[:-1]
        assistant_prefix = messages[-1]['content']

        if prompt_messages and prompt_messages[0]['role'] == 'system' and prompt_messages[0]['content'] == "":
            prompt_messages = prompt_messages[1:]

        rendered_prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {'text': rendered_prompt + assistant_prefix}

    def _string_formatter_completion_only_llama3_base(self, example):
        if 'messages' not in example:
            raise ValueError("No messages in the example")

        messages = example['messages']
        if len(messages) == 0:
            raise ValueError("No messages in the example")
        if messages[-1]['role'] != 'assistant':
            raise ValueError("completion only mode should end with a header of assistant message")

        parts = []
        pt = 0
        if messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            pt = 1
            if system_prompt:
                parts.append(f"System: {system_prompt}\n\n")

        while pt < len(messages) - 1:
            if messages[pt]['role'] != 'user':
                raise ValueError("the message should be user - assistant alternation")
            parts.append(f"User: {messages[pt]['content']}\n")
            pt += 1

            if pt >= len(messages) - 1:
                break
            if messages[pt]['role'] != 'assistant':
                raise ValueError("the message should be user - assistant alternation")
            parts.append(f"Assistant: {messages[pt]['content']}\n")
            pt += 1

        parts.append(f"Assistant: {messages[-1]['content']}")
        return {'text': ''.join(parts)}

    def __call__(self, text, max_new_tokens=1024,
                 do_sample=True, top_p=0.9, temperature=0.6, use_cache=True, top_k=50,
                 repetition_penalty=1.0, length_penalty=1.0, k=None, **kwargs):

        self.update_conversation(user_message=text)
        _, tokens_input = self.prepare_model_input(conversation=self.conversation, max_new_tokens=max_new_tokens)

        model_for_generation = self._get_model_for_generation()
        model_device = self._get_model_device()
        tokens_input = tokens_input.unsqueeze(0).to(model_device)
        input_length = tokens_input.shape[1]
        k = self._validate_distribution_tracking_steps(k)
        generation_kwargs = dict(
            input_ids=tokens_input,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            stopping_criteria=self.stopping_criteria,
        )
        generation_kwargs.update(kwargs)
        if k is not None:
            generation_kwargs["return_dict_in_generate"] = True
            generation_kwargs["output_scores"] = True

        outputs = model_for_generation.generate(**generation_kwargs)
        sequences = outputs.sequences if k is not None else outputs

        output_text = self.tokenizer.decode(sequences[0][input_length:], skip_special_tokens=True)
        self.update_conversation(assistant_message=output_text)

        if k is not None:
            tracked = self._extract_tracked_token_distributions(outputs, input_length, k)
            return {
                "output_text": output_text,
                "generated_token_ids": tracked["generated_token_ids"][0],
                "token_distributions": tracked["token_distributions"][0],
                "tracked_steps": tracked["tracked_steps"],
            }

        return output_text

    def generate_one_shot(self, input, max_new_tokens=1024,
                          do_sample=True, top_p=0.9, temperature=0.6, use_cache=True, top_k=50,
                          repetition_penalty=1.0, length_penalty=1.0, k=None, **kwargs):
        # a one-shot conversation, input can be a string, a list of messages, or a dictionary with 'messages' key
        # no history will be maintained for one-shot conversation

        if isinstance(input, dict) or isinstance(input, list):
            input = self.validate_conversation(input)
        elif isinstance(input, str):
            input = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
        else:
            raise ValueError(f"input {input} is not a valid conversation input")

        _, tokens_input = self.prepare_model_input(input, max_new_tokens)
        model_for_generation = self._get_model_for_generation()
        model_device = self._get_model_device()
        tokens_input = tokens_input.to(model_device)
        input_length = tokens_input.shape[1]
        k = self._validate_distribution_tracking_steps(k)

        generation_kwargs = dict(
            input_ids=tokens_input,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            stopping_criteria=self.stopping_criteria,
        )
        generation_kwargs.update(kwargs)
        if k is not None:
            generation_kwargs["return_dict_in_generate"] = True
            generation_kwargs["output_scores"] = True

        outputs = model_for_generation.generate(**generation_kwargs)
        sequences = outputs.sequences if k is not None else outputs

        output_text = self.tokenizer.decode(sequences[0][input_length:], skip_special_tokens=True)  # the model output part
        full_text = self.tokenizer.decode(sequences[0], skip_special_tokens=True)  # the whole conversation

        if k is not None:
            tracked = self._extract_tracked_token_distributions(outputs, input_length, k)
            return {
                "output_text": output_text,
                "full_text": full_text,
                "generated_token_ids": tracked["generated_token_ids"][0],
                "token_distributions": tracked["token_distributions"][0],
                "tracked_steps": tracked["tracked_steps"],
            }

        return output_text, full_text

    def generate_one_shot_in_batch(self, inputs, accelerator, max_new_tokens=1024,
                                   do_sample=True, top_p=0.9, temperature=0.6, use_cache=True, top_k=50,
                                   repetition_penalty=1.0, length_penalty=1.0, k=None, **kwargs):
        # a one-shot conversation, input can be a string, a list of messages, or a dictionary with 'messages' key
        # no history will be maintained for one-shot conversation
        # this function is for batch inference to accelerate the evaluation

        inputs_processed = []

        for item in inputs:
            if isinstance(item, dict) or isinstance(item, list):
                item_processed = self.validate_conversation(item)
            elif isinstance(item, str):
                item_processed = self.init_conversation() + [{'role': 'user', 'content': item}, {'role': 'assistant', 'content': ''}]
            else:
                raise ValueError(f"input {item} is not a valid conversation input")

            item_processed = self.string_formatter({'messages': item_processed})['text']
            inputs_processed.append(item_processed)

        model_for_generation = self._get_model_for_generation()
        model_device = self._get_model_device()
        model_inputs = self.tokenizer(inputs_processed, padding=True, return_tensors="pt").to(model_device)
        k = self._validate_distribution_tracking_steps(k)

        generation_kwargs = dict(
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            stopping_criteria=self.stopping_criteria,
        )
        generation_kwargs.update(kwargs)
        if k is not None:
            generation_kwargs["return_dict_in_generate"] = True
            generation_kwargs["output_scores"] = True

        outputs = model_for_generation.generate(**generation_kwargs)
        sequences = outputs.sequences if k is not None else outputs

        full_texts = []  # the whole conversation texts
        output_texts = []  # the model output part texts

        for i, item in enumerate(sequences):
            input_pos = model_inputs['attention_mask'][i].nonzero()

            input_length = input_pos.shape[0]  # how many input tokens
            start_pos = input_pos[0][0]  # the first non-padding token

            full_text = self.tokenizer.decode(item, skip_special_tokens=True)
            output_text = self.tokenizer.decode(item[start_pos + input_length:], skip_special_tokens=True)

            full_texts.append(full_text)
            output_texts.append(output_text)

        if k is not None:
            tracked = self._extract_tracked_token_distributions(outputs, model_inputs['input_ids'].shape[1], k)
            return {
                "output_texts": output_texts,
                "full_texts": full_texts,
                "generated_token_ids": tracked["generated_token_ids"],
                "token_distributions": tracked["token_distributions"],
                "tracked_steps": tracked["tracked_steps"],
            }

        return output_texts, full_texts

    def validate_conversation(self, conversation=None):
        # validate the conversation format, return the conversation in OpenAI chat format

        if conversation is None:
            conversation = self.conversation

        if isinstance(conversation, dict):
            if 'messages' not in conversation:
                raise ValueError(f"conversation {conversation} does not have 'messages' key")
            convs = conversation['messages']
        else:
            convs = conversation

        if not isinstance(convs, list):
            raise ValueError(f"conversation {conversation} is not a valid list of messages")

        if len(convs) == 0:
            raise ValueError(f"the conversation {conversation} is empty")

        for conv in convs:
            if 'role' not in conv or 'content' not in conv:
                raise ValueError(f"the message {conv} does not have 'role' or 'content' key")

        if convs[0]['role'] != 'system':
            convs = self.init_conversation() + convs

        pt = 1

        while pt < len(convs):
            if convs[pt]['role'] != 'user':
                raise ValueError(f"the message should be user - assistant alternation, but the {pt}th message is {convs[pt]['role']}")
            pt += 1
            if pt >= len(convs):
                break
            if convs[pt]['role'] != 'assistant':
                raise ValueError(f"the message should be user - assistant alternation, but the {pt}th message is {convs[pt]['role']}")
            pt += 1
        return convs

    def init_conversation(self, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        return [{'role': 'system', 'content': system_prompt}]

    def refresh_conversation(self):
        self.conversation = self.init_conversation()

    def update_conversation(self, conversation=None, user_message=None, assistant_message=None):
        if conversation is None:
            conversation = self.conversation

        if user_message is None and assistant_message is None:
            raise ValueError("user_message or assistant_message should be provided")

        if user_message is not None:
            if conversation[-1]['role'] == 'user':
                raise ValueError("the message should be user - assistant alternation")
            conversation.append({'role': 'user', 'content': user_message})

        if assistant_message is not None:
            if conversation[-1]['role'] == 'assistant' or conversation[-1]['role'] == 'system':
                raise ValueError("the message should be user - assistant alternation")
            conversation.append({'role': 'assistant', 'content': assistant_message})

    def prepare_model_input(self, conversation=None, max_new_tokens=512):
        if conversation is None:
            conversation = self.conversation
        string_input = self.string_formatter({'messages': conversation})['text']

        tokens_input = self.tokenizer.encode(
            string_input,
            return_tensors="pt",
            max_length=self.max_length - max_new_tokens,
            truncation=True,
        )

        return string_input, tokens_input
