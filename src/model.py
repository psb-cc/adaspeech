"""
Model Module
    This module contains the SpeechEncoder2Adapter2Llm class, which is the main model class for the ASR task.
    Do not change the class defintion to add new speech encoders, llms or adapters.
    Instead, create a new class (and register) in the speech_encoder.py, llm.py and adapter.py.
"""

import torch
from math import sqrt
import os

from src.adapter import get_adapter_class
from src.llm import LLM, get_llm_class
from src.speech_encoder import SpeechEncoder, get_speech_encoder_class
from src.utils import compute_accuracy, get_model_size


class SpeechEncoder2Adapter2Llm(torch.nn.Module):
    def __init__(
        self,
        speech_encoder_name: str,
        llm_name: str,
        adapter_name: str,
        device="cuda:0",
        path_to_adapter_weights=None,
        lora_config_dict=None,
        lora_adapter_ckpt_path=None,
    ):
        super().__init__()
        self.device = device

        # Initialize components using abstract methods
        self.speech_encoder: SpeechEncoder = get_speech_encoder_class(
            speech_encoder_name
        )()
        self.llm: LLM = get_llm_class(llm_name)(device=device)
        self.tokenizer = self.llm.tokenizer

        # Get dimensions
        self.speech_encoder_dim = self.speech_encoder.get_embedding_dim()
        self.llm_dim = self.llm.get_embedding_dim()

        # Initialize adapter using abstract method
        adapter_class = get_adapter_class(adapter_name)
        
        # just make sure if the speech encoder has get_num_blocks then adapter must have _add_auxilliary_input_conditioning_layers method
        if hasattr(self.speech_encoder, "get_num_blocks"):
            if not hasattr(adapter_class, "_add_auxilliary_input_conditioning_layers"):
                raise ValueError("Expected adapter to have _add_auxilliary_input_conditioning_layers method")
            else:
                self.adapter = adapter_class(
                    speech_encoder_dim=self.speech_encoder_dim,
                    llm_dim=self.llm_dim,
                    path_to_adapter_weights=path_to_adapter_weights,
                    num_blocks=self.speech_encoder.get_num_blocks(),
                    num_mels=self.speech_encoder.get_num_mels()
                )
        else:
            self.adapter = adapter_class(
                speech_encoder_dim=self.speech_encoder_dim,
                llm_dim=self.llm_dim,
                path_to_adapter_weights=path_to_adapter_weights,
            )

        print("number of trainable parameters in adapter", get_model_size(self.adapter))

        # Move components to device
        self.speech_encoder.to(device)
        self.adapter.to(device)

        # LoRA activation and loading
        if lora_config_dict is not None:
            self._init_lora_from_dict(lora_config_dict)
            if lora_adapter_ckpt_path is not None:
                print(f"Loading LoRA adapter weights from {lora_adapter_ckpt_path}")
                self.load_lora_adapter(lora_adapter_ckpt_path)
            print("number of trainable parameters in LoRA adapter", sum(p.numel() for p in self.llm.model.parameters() if p.requires_grad))

    def _init_lora_from_dict(self, lora_cfg):
        from peft import LoraConfig, TaskType
        task_type = getattr(TaskType, lora_cfg.get("task_type", "CAUSAL_LM"))
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=task_type,
        )
        self.activate_llm_lora(lora_config)

    def forward(self, batch, inference=False, use_matching_loss=False):
        audio = batch.get("audio", None)
        audio_mask = batch.get("audio_mask", None)
        audio_mel = batch.get("audio_mel", None)
        audio_mel_mask = batch.get("audio_mel_mask", None)
        encoder_attention_mask = None

        if audio is not None:
            audio = audio.to(self.device)
        if audio_mask is not None:
            audio_mask = audio_mask.to(self.device)
            audio_padding_mask = 1 - audio_mask
            audio_mel_padding_mask = None
            encoder_attention_mask = audio_mask

        if audio_mel is not None:
            audio_mel = audio_mel.to(self.device)
        if audio_mel_mask is not None:
            audio_mel_mask = audio_mel_mask.to(self.device)
            audio_mel_padding_mask = 1 - audio_mel_mask
            audio_padding_mask = None
            encoder_attention_mask = audio_mel_mask

        modality_mask = batch["modality_mask"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        if "labels" in batch:
            labels = batch["labels"].to(self.device)
        else:
            labels = None

        # [bs, max_num_samples] --> [bs, max_num_samples, speech_encoder_dim]
        speech_encoder_output = self.speech_encoder.encode_speech(
            audio=audio,
            audio_padding_mask=audio_padding_mask,
            audio_mel=audio_mel,
            audio_mel_padding_mask=audio_mel_padding_mask,
        )
        # print("speech encoder input", audio.shape, audio_padding_mask.shape)
        # print("speech encoder output", speech_encoder_output.shape)

        if hasattr(self.adapter, "_add_auxilliary_input_conditioning_layers"):
            adapter_output = self.adapter.project(speech_encoder_output, audio_mel)
        else:
            adapter_output = self.adapter.project(speech_encoder_output)

        if not inference:
            if use_matching_loss:
                # for matching loss, we need to get just the label_ids from the labels tensor with only text side padding
                # e.g. [..., -100, -100, -100, 72511, 15934, 12653, 10688, 2, -100, -100,-100, -100, -100, -100, -100, -100, -100, -100],
                # keep:[..., no,   no,   no,   yes,  yes,  yes,  yes,  yes,  yes,  yes,  yes,  yes,  yes,  yes,  yes,  yes,  yes,  yes],

                longest_label_len = labels.ge(-1).sum(dim=1).max()  # longest label length
                padded_labels_ids = labels[:, -longest_label_len:]
                # print("padded labels ids", padded_labels_ids)
                # print("input_ids", input_ids[:, -longest_label_len:])
                padded_labels_attention_mask = padded_labels_ids.ge(-1)
                padded_labels_ids[~padded_labels_attention_mask] = self.tokenizer.pad_token_id
                # print("padded labels attention mask", padded_labels_attention_mask.sum(dim=1))

                labels_embeds = self.llm.embed_tokens(padded_labels_ids) # [bs, max_num_tokens_this_batch, llm_dim]
                # print("labels embeds", labels_embeds.shape)
                # print("adapter output", adapter_output.shape)

                # now get SDPA attention weights and transforme the adapter output to the same shape as labels_embeds
                attention_weights = torch.bmm(labels_embeds, adapter_output.transpose(1, 2)) / sqrt(self.llm_dim)
                attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)  # over keys: adapter_output

                # Optional: zero attention for padded labels (queries)
                attention_weights = attention_weights * padded_labels_attention_mask[:, :, None]
                # print("attention weights", attention_weights.shape)

                aux_adapter_output = torch.bmm(attention_weights, adapter_output) # [bs, max_num_tokens_this_batch, llm_dim]
                # print("aux adapter output", aux_adapter_output.shape)

                # calculate the cosine and mse loss and add them together
                cosine_sim = torch.nn.functional.cosine_similarity(aux_adapter_output, labels_embeds, dim=-1) # [bs, max_num_tokens_this_batch]
                cosine_loss = 1 - (cosine_sim.sum() / padded_labels_attention_mask.sum())
                # print("cosine loss", cosine_loss)

                mse_loss = torch.nn.functional.mse_loss(aux_adapter_output[padded_labels_attention_mask], labels_embeds[padded_labels_attention_mask])
                # print("mse loss", mse_loss)

                matching_loss = 0.1 * mse_loss + 0.4 * cosine_loss
                # print("matching loss", matching_loss)
            else:
                # NOTE: sloppy function signature ik; kept for backward compatibility
                matching_loss = torch.tensor(0.0, device=self.device)

        # print("adapter output", adapter_output.shape)
        # NOTE: in collator, we added -1s for all speech token positions, make all -1s to 0s so that tokenizer doesnt freak out
        input_ids[input_ids == -1] = self.tokenizer.pad_token_id

        # [bs, max_num_tokens] --> [bs, max_num_tokens, llm_dim]
        input_embeds = self.llm.embed_tokens(input_ids)
        input_embeds = input_embeds.to(self.device)

        # print("input ids", input_ids.shape)
        # print("input embeds", input_embeds.shape)
        # print("input embeds sum", input_embeds.sum())

        # NOTE: now we need to plug the adapter output in the input_embeds at the speech token positions; it should slot right in
        ## get the start of the speech token positions from the modality mask
        modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
        # print("modality mask start indices", modality_mask_start_indices)
        ## get the length of the speech tokens from the modality mask
        modality_lengths = torch.clamp(
            modality_mask.sum(dim=1), max=adapter_output.shape[1]
        ).tolist()
        # print("modality lengths", modality_lengths)
        ## adapter_output_pad is a tensor of zeros with the same shape as input_embeds; it will contain adapter_output at the speech token positions
        adapter_output_pad = torch.zeros_like(input_embeds)
        # print("adapter output pad sum", adapter_output_pad.sum())
        ## plug the adapter output in the input_embeds at the speech token positions
        for i in range(adapter_output.shape[0]):
            adapter_output_pad[
                i,
                modality_mask_start_indices[i] : modality_mask_start_indices[i]
                + modality_lengths[i],
            ] = adapter_output[i][: modality_lengths[i]]
        # print("adapter output pad sum", adapter_output_pad.sum())
        ## finally, add the adapter output to the input_embeds at the speech token positions; slots right in
        input_embeds = adapter_output_pad + input_embeds * (~modality_mask[:, :, None])
        # print("input embeds sum", input_embeds.sum())

        # pass the input_embeds to the llm
        model_outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        # print("model outputs", model_outputs)

        if inference:
            return input_embeds, attention_mask

        # calculate the token prediction accuracy
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                labels.detach()[:, 1:],
                ignore_label=-100,
            )

        return model_outputs, matching_loss, acc

    def generate(self, batch):
        input_embeds, attention_mask = self.forward(batch, inference=True)

        model_outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=200,
            num_beams=4,
            do_sample=False,
            min_length=1,
            top_p=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=1.0,
        )
        return model_outputs

    def save_lora_adapter(self, path):
        from peft import get_peft_model_state_dict
        torch.save(get_peft_model_state_dict(self.llm.model), path)

    def load_lora_adapter(self, path):
        print(f"Loading LoRA adapter weights from {path}")
        from peft import set_peft_model_state_dict
        peft_model_state_dict = torch.load(path)
        set_peft_model_state_dict(self.llm.model, peft_model_state_dict)

    def activate_llm_lora(self, lora_config):
        self.llm.activate_lora(lora_config)
