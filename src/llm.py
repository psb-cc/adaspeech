"""
LLM Module

To Extend:
    1. Make a new class subclassing `LLM` with the following methods:
        - `__init__`: Initialize the LLM
        - `get_embedding_dim`: Get the dimension of the embedding
        - `embed_tokens`: Embed the tokens
        - [optional] `generate`: Generate the text
"""

from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

LLM_REGISTRY = {}


def register_llm(name):
    """
    Decorator to register LLM classes in the LLM_REGISTRY dictionary.

    Args:
        name (str): The name of the LLM.
    """

    def wrapper(cls):
        LLM_REGISTRY[name] = cls
        return cls

    return wrapper


def get_llm_class(name):
    """Get LLM class by name from registry"""
    if name not in LLM_REGISTRY:
        available_llms = list(LLM_REGISTRY.keys())
        raise ValueError(
            f"Unknown LLM class: {name}. Available options: {available_llms}"
        )
    return LLM_REGISTRY[name]


class LLM(ABC, torch.nn.Module):
    def __init__(
        self,
        path_to_pretrained_model=None,
        freeze_layers=True,
        device="cuda:0",
    ):
        super().__init__()
        if path_to_pretrained_model is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                path_to_pretrained_model,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                local_files_only=False,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                path_to_pretrained_model,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=False,
            )
            if freeze_layers:
                for _, param in self.model.named_parameters():
                    param.requires_grad = False
            self.model.eval()
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.is_lora_active = False

    def activate_lora(self, lora_config=None):
        if lora_config is None:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        self.model = get_peft_model(self.model, lora_config)
        self.is_lora_active = True
        print("LoRA adapters activated.")

    def forward(self, inputs_embeds, attention_mask, labels):
        return self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

    @abstractmethod
    def embed_tokens(self, batch_of_input_ids):
        pass

    @abstractmethod
    def get_embedding_dim(self):
        pass

    def generate(
        self,
        inputs_embeds,
        attention_mask,
        max_new_tokens=200,
        num_beams=20,
        do_sample=False,
        min_length=1,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
    ):
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )


@register_llm("krutrim-2-12b-instruct")
class Krutrim_2_12B_Instruct(LLM):
    def __init__(
        self,
        path_to_pretrained_model="krutrim-ai-labs/Krutrim-2-instruct",
        freeze_layers=True,
        device="cuda:0",
    ):
        super().__init__(path_to_pretrained_model, freeze_layers, device)

    def get_embedding_dim(self):
        return self.model.model.embed_tokens.weight.shape[1]

    def embed_tokens(self, batch_of_input_ids):
        batch_of_input_ids = batch_of_input_ids
        if self.is_lora_active:
            return self.model.base_model.model.model.embed_tokens(batch_of_input_ids)
        else:
            return self.model.model.embed_tokens(batch_of_input_ids)
