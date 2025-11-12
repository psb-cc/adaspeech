"""
Speech Encoder Module

To Extend:
    - Make a new class subclassing `SpeechEncoder` with the following methods:
        - `__init__`: Initialize the speech encoder
        - `encode_speech`: Encode the speech
        - `get_embedding_dim`: Get the dimension of the embedding
"""

from abc import ABC, abstractmethod

import torch
from transformers import AutoModel

SPEECH_ENCODER_REGISTRY = {}


def register_speech_encoder(name):
    """
    Decorator to register speech encoder classes in the SPEECH_ENCODER_REGISTRY dictionary.

    Args:
        name (str): The name of the speech encoder.
    """

    def wrapper(cls):
        SPEECH_ENCODER_REGISTRY[name] = cls
        return cls

    return wrapper


class SpeechEncoder(ABC, torch.nn.Module):
    def __init__(self, model_name: str = None, freeze_layers: bool = True):
        super().__init__()
        self.speech_encoder = None
        if model_name is not None:
            self._initialize_encoder(model_name)
            if freeze_layers:
                self._freeze_encoder()

    def _initialize_encoder(self, model_name: str):
        """Initialize the speech encoder model. Override this method for custom initialization."""
        self.speech_encoder = AutoModel.from_pretrained(model_name)

    def _freeze_encoder(self):
        """Freeze the encoder parameters if they exist."""
        if self.speech_encoder is not None:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False

    @abstractmethod
    def encode_speech(
        self,
        audio=None,
        audio_padding_mask=None,
        audio_mel=None,
        audio_mel_padding_mask=None,
    ):
        pass

    @abstractmethod
    def get_embedding_dim(self):
        pass


def get_speech_encoder_class(name):
    """Get speech encoder class by name from registry"""
    if name not in SPEECH_ENCODER_REGISTRY:
        available_speech_encoders = list(SPEECH_ENCODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown speech encoder class: {name}. Available options: {available_speech_encoders}"
        )
    return SPEECH_ENCODER_REGISTRY[name]



@register_speech_encoder("openai-whisper-large-v3")
class OpenaiWhisperLargeV3SpeechEncoder(SpeechEncoder):
    def __init__(self, model_name="openai/whisper-large-v3", freeze_layers=True):
        super().__init__(model_name, freeze_layers)
        self.speech_encoder = self.speech_encoder.encoder

    def get_embedding_dim(self):
        return 1280

    def encode_speech(
        self, audio_mel, audio_mel_padding_mask, audio=None, audio_padding_mask=None
    ):
        speech_encoder_output = self.speech_encoder(
            input_features=audio_mel, attention_mask=audio_mel_padding_mask
        )
        return speech_encoder_output[0]