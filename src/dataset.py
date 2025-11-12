import copy
import json

import numpy as np
import torch
import whisper


class AsrDataset(torch.utils.data.Dataset):
    """
    A dataset class for speech data stored in JSONL files.
    Processes audio files and their corresponding text for speech recognition tasks.
    """

    def __init__(
        self,
        tokenizer=None,
        prompt=None,
        mel_size=80,
        fix_length_audio=-1,
        fix_audio_duration=-1,
        inference_mode=False,
        normalize=False,
        input_type="raw",
        target_column="target",
        path_to_jsonl_file=None,
    ):
        """
        Initialize the speech dataset.

        Args:
            dataset_config: Configuration dictionary for the dataset
            tokenizer: Tokenizer to convert text to/from tokens
            split: Dataset split to use ('train' or other, typically 'val')
        """
        super().__init__()
        self.tokenizer = tokenizer

        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = prompt
        self.mel_size = mel_size
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = fix_length_audio
        self.fix_audio_duration = fix_audio_duration
        self.inference_mode = inference_mode
        self.normalize = normalize
        self.input_type = input_type
        self.target_column = target_column
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]"

        self.data_list = []
        with open(path_to_jsonl_file, encoding="utf-8") as fin:
            for line in fin:
                try:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
                except:
                    continue

    def get_source_len(self, data_dict):
        """Get the length of the source (audio) data."""
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
        """Get the length of the target (text) data."""
        return data_dict["target_len"] if "target_len" in data_dict else 0

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Args:
            index: Index of the sample to retrieve

        Returns:
            A dictionary containing the processed sample data
        """
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get(self.target_column, None)
        task = data_dict.get("prompt", "ASR")
        key = data_dict.get("key", None)

        audio_raw = whisper.load_audio(audio_path)

        audio_duration = (
            len(audio_raw) / 16000
        )  # Duration in seconds assuming 16kHz sampling rate

        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320  # Downsampling factor for fairseq
            audio_length = audio_length // 5  # Additional downsampling from FC layers
        elif self.input_type == "mel":
            # print("debug: ", self.fix_audio_duration, "audio_raw.shape", audio_raw.shape)
            if (
                self.fix_audio_duration > 0
            ):  # this is to ensure that all whisper based models need audios with fixed duration of 30 seconds
                # print("audio_raw.shape", audio_raw.shape)
                max_samples = int(
                    self.fix_audio_duration * 16000
                )  # 16kHz sampling rate
                audio_raw = whisper.pad_or_trim(audio_raw, max_samples)
                # print("audio_raw.shape after padding", audio_raw.shape)

            audio_mel = whisper.log_mel_spectrogram(
                audio_raw, n_mels=self.mel_size
            ).permute(1, 0)
            # print("audio_mel.shape", audio_mel.shape)
            audio_length = (audio_mel.shape[0] + 1) // 2  # Whisper downsample factor
            audio_length = audio_length // 5  # Additional downsampling
            # print("audio_length after downsampling", audio_length)

        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio

        audio_pseudo = torch.full((audio_length,), -1)  # Placeholder for audio features

        prompt = self.prompt
        if prompt is None:
            prompt = "Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. "
        prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)

        if self.inference_mode:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))
            example_mask = example_ids.ge(-1)

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
                "audio_duration": audio_duration,
            }

        answer = self.answer_template.format(target)
        example = prompt + answer
        example_ids = self.tokenizer.encode(example)
        example_ids.append(self.tokenizer.eos_token_id)
        example_ids = torch.tensor(example_ids, dtype=torch.int64)
        example_ids = torch.cat((audio_pseudo, example_ids))

        labels_ids = copy.deepcopy(example_ids)
        labels_ids[: audio_length + prompt_length] = -1

        example_mask = example_ids.ge(-1)
        label_mask = labels_ids.ge(0)

        example_ids[~example_mask] = 0
        labels_ids[~label_mask] = self.IGNORE_INDEX

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
            "audio_duration": audio_duration,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        """
        Pad a sequence to the specified length.

        Args:
            sequence: The sequence to pad (can be list, tensor, or numpy array)
            max_length: The desired length
            padding_idx: The value to use for padding

        Returns:
            The padded sequence
        """
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (
                        sequence,
                        torch.full(
                            ([max_length - len(sequence)] + list(sequence.size())[1:]),
                            padding_idx,
                        ),
                    )
                )
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (
                        sequence,
                        np.full(
                            (max_length - len(sequence),) + sequence.shape[1:],
                            padding_idx,
                        ),
                    )
                )
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        """
        Pad a sequence with a specified number of padding elements.

        Args:
            sequence: The sequence to pad
            padding_length: Number of padding elements to add (if positive) or trim (if negative)
            padding_idx: The value to use for padding
            padding_side: Which side to add padding ('left' or 'right')

        Returns:
            The padded sequence
        """
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat(
                            (
                                torch.full(
                                    ([padding_length] + list(sequence.size())[1:]),
                                    padding_idx,
                                ),
                                sequence,
                            )
                        )
                    else:
                        sequence = torch.cat(
                            (
                                sequence,
                                torch.full(
                                    ([padding_length] + list(sequence.size())[1:]),
                                    padding_idx,
                                ),
                            )
                        )
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (
                        sequence,
                        np.full((padding_length,) + sequence.shape[1:], padding_idx),
                    )
                )
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        """
        Collate function for batching samples.
        Handles padding and alignment of variable-length sequences.

        Args:
            samples: List of samples to batch

        Returns:
            Dictionary containing the batched data
        """
        assert samples is not None

        input_prompt_lengths = [s["audio_length"] + s["prompt_length"] for s in samples]
        input_answer_lengths = [
            len(s["input_ids"]) - s["audio_length"] - s["prompt_length"]
            for s in samples
        ]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)

        input_ids = torch.stack(
            [
                self.padding(
                    self.padding(
                        samples[index]["input_ids"],
                        input_prompt_max_length - input_prompt_lengths[index],
                        self.tokenizer.pad_token_id,
                        padding_side="left",
                    ),
                    input_answer_max_length - input_answer_lengths[index],
                    self.tokenizer.pad_token_id,
                )
                for index in range(len(samples))
            ]
        )

        attention_mask = torch.stack(
            [
                self.padding(
                    self.padding(
                        samples[index]["attention_mask"],
                        input_prompt_max_length - input_prompt_lengths[index],
                        False,
                        padding_side="left",
                    ),
                    input_answer_max_length - input_answer_lengths[index],
                    False,
                )
                for index in range(len(samples))
            ]
        )

        if self.input_type == "raw":
            audio_raw_max_length = max([s["audio"].shape[0] for s in samples])
            audio_raw = torch.stack(
                [self.pad(s["audio"], audio_raw_max_length, 0) for s in samples]
            )
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, : sample["audio"].shape[0]] = 1
        elif self.input_type == "mel":
            audio_mel_max_length = max([s["audio_mel"].shape[0] for s in samples])
            audio_mel = torch.stack(
                [self.pad(s["audio_mel"], audio_mel_max_length, 0) for s in samples]
            )
            audio_mel_mask = torch.zeros(len(samples), audio_mel_max_length)
            for line, sample in enumerate(samples):
                audio_mel_mask[line, : sample["audio_mel"].shape[0]] = 1

        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index]
            modality_mask[
                index, padding_left : padding_left + samples[index]["audio_length"]
            ] = True

        # Get audio durations
        audio_durations = torch.tensor([s["audio_duration"] for s in samples])

        # permute from [batch, time, mel] to [batch, mel, time]
        # print("audio_mel.shape", audio_mel.shape)
        # print("audio_mel_mask.shape", audio_mel_mask.shape)

        if self.input_type == "mel":
            audio_mel = audio_mel.permute(0, 2, 1)

        # print("audio_mel.shape after permute", audio_mel.shape)
        # print("audio_mel_mask.shape after permute", audio_mel_mask.shape)

        if self.inference_mode:
            keys = [s["key"] for s in samples]
            targets = [s["target"] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_mask": (
                    audio_mel_mask if self.input_type == "mel" else None
                ),
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets,
                "audio_duration": audio_durations,
            }

        labels = torch.stack(
            [
                self.padding(
                    self.padding(
                        samples[index]["labels"],
                        input_prompt_max_length - input_prompt_lengths[index],
                        self.IGNORE_INDEX,
                        padding_side="left",
                    ),
                    input_answer_max_length - input_answer_lengths[index],
                    self.IGNORE_INDEX,
                )
                for index in range(len(samples))
            ]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_mask": (audio_mel_mask if self.input_type == "mel" else None),
            "modality_mask": modality_mask,
            "audio_duration": audio_durations,
        }
