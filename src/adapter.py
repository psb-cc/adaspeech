"""
Adapter Module

To Extend:
    1. Make a new class subclassing `Adapter` with the following methods:
        - `__init__`: Initialize the adapter
        - `forward`: Forward pass
        - `project`: Project the output of the adapter to the LLM input dimension
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import Blip2QFormerConfig, Blip2QFormerModel

ADAPTER_REGISTRY = {}


def register_adapter(name):
    """
    Decorator to register adapter classes in the ADAPTER_REGISTRY dictionary.

    Args:
        name (str): The name of the adapter.
    """

    def wrapper(cls):
        ADAPTER_REGISTRY[name] = cls
        return cls

    return wrapper


def get_adapter_class(name):
    """Get adapter class by name from registry"""
    if name not in ADAPTER_REGISTRY:
        available_adapters = list(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown adapter class: {name}. Available options: {available_adapters}"
        )
    return ADAPTER_REGISTRY[name]


class Adapter(ABC, nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @staticmethod
    def downsample_speech_features(x, k):
        """
        Downsample speech features by factor k
        Input: [bs, L, speech_enc_dim]
        Output: [bs, L // k, speech_enc_dim * k]
        """
        batch_size, seq_len, dim = x.size()
        # print(f"Encoder output shape: {x.shape}")

        # Discard frames that don't fit evenly into downsampling
        num_frames_to_discard = seq_len % k
        # print(f"Num frames to discard: {num_frames_to_discard}")
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        # print(f"Encoder output shape after discarding frames: {x.shape}")

        seq_len = x.size(1)
        x = x.contiguous()
        # print(f"Encoder output shape after contiguous: {x.shape}")

        # Reshape to downsample: combine k consecutive frames
        x = x.view(batch_size, seq_len // k, dim * k)
        # print(f"Encoder output shape after downsampling: {x.shape}")

        return x

    @abstractmethod
    def project(self, x):
        pass


@register_adapter("multi-res-conv-small-projector")
class MultiResConvSmallProjector(Adapter):
    def __init__(
        self,
        speech_encoder_dim=1024,
        llm_dim=2048,
        downsampling_rate=5,
        path_to_adapter_weights=None,
    ):
        super().__init__()
        self.k = downsampling_rate
        self.speech_encoder_dim = speech_encoder_dim
        self.llm_dim = llm_dim

        # 3 convs at different resolutions
        self.conv1 = nn.Conv1d(
            speech_encoder_dim, 64, kernel_size=3, stride=2, padding=1
        )
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.SiLU()
        self.conv2 = nn.Conv1d(
            speech_encoder_dim, 64, kernel_size=5, stride=2, padding=2
        )
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.relu2 = nn.SiLU()
        self.conv3 = nn.Conv1d(
            speech_encoder_dim, 64, kernel_size=7, stride=2, padding=3
        )
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.relu3 = nn.SiLU()

        # final conv to project to llm_dim
        self.conv_final = nn.Conv1d(
            64 * 3, llm_dim, kernel_size=1, stride=1, padding=1
        )
        self.batch_norm_final = nn.BatchNorm1d(llm_dim)
        self.relu_final = nn.SiLU()

        if path_to_adapter_weights is not None:
            self.load_state_dict(torch.load(path_to_adapter_weights)["model"])

    def forward(self, x):
        x = x.transpose(1, 2)  # [bs, speech_enc_dim * k, L]
        conv1_out = self.conv1(x)  # [bs, 32, L]
        conv1_out = self.batch_norm1(conv1_out)  # [bs, 32, L]
        conv1_out = self.relu1(conv1_out)  # [bs, 32, L]

        conv2_out = self.conv2(x)  # [bs, 32, L]
        conv2_out = self.batch_norm2(conv2_out)  # [bs, 32, L]
        conv2_out = self.relu2(conv2_out)  # [bs, 32, L]

        conv3_out = self.conv3(x)  # [bs, 32, L]
        conv3_out = self.batch_norm3(conv3_out)  # [bs, 32, L]
        conv3_out = self.relu3(conv3_out)  # [bs, 32, L]

        all_conv_out = torch.cat(
            [conv1_out, conv2_out, conv3_out], dim=1
        )  # [bs, 192, L]
        x = self.conv_final(all_conv_out)  # [bs, llm_dim, L]
        x = self.batch_norm_final(x)  # [bs, llm_dim, L]
        x = self.relu_final(x)  # [bs, llm_dim, L]

        x = x.transpose(1, 2)  # [bs, L, llm_dim]
        return x

    def project(self, x):
        return self.forward(x)