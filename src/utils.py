import json
import logging
import os
import random

import numpy as np
import torch
import yaml
from jiwer import wer


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_random_seed(seed=69):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6


def batch_to_device(batch, device):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()  # (FIX:MZY):return torch.Tensor type


def save_model(model, optimizer, path_to_save_model_checkpoint):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path_to_save_model_checkpoint,
    )


def calculate_rtfx_score(start_time, end_time, audio_length):
    """Calculate RTFX score between start time and end time.

    Args:
        start_time: Start time of inference
        end_time: End time of inference
        audio_length: List of audio lengths in seconds

    Returns:
        float: RTFx score (sum of audio lengths / total inference time)
    """
    total_audio_length = sum(audio_length)
    total_inference_time = end_time - start_time
    return total_audio_length / total_inference_time


def save_rtfx_score_to_consolidated_file(
    rtfx_score: dict, adapter_dir: str, adapter_base: str
):
    """Save RTFX score to a consolidated JSON file that contains all data cuts."""
    logger = logging.getLogger(__name__)
    consolidated_file = os.path.join(
        adapter_dir, f"{adapter_base}_all_rtfx_results.json"
    )
    existing_results = {}
    if os.path.exists(consolidated_file):
        try:
            with open(consolidated_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(
                f"Could not load existing results from {consolidated_file}, starting fresh"
            )
            existing_results = {}

    existing_results.update(rtfx_score)

    with open(consolidated_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

    logger.info(f"RTFX results saved to consolidated file: {consolidated_file}")
    return consolidated_file


def calculate_wer_score(ground_truth, prediction):
    """Calculate WER score between ground truth and prediction."""
    return wer(ground_truth, prediction)


def read_and_calculate_wer(gt_file_path, pred_file_path, test_data_name):
    """Read ground truth and prediction files, calculate WER, and return results."""
    logger = logging.getLogger(__name__)
    gt_texts = []
    pred_texts = []

    # Read ground truth file
    with open(gt_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                gt_texts.append(parts[1])

    # Read prediction file
    with open(pred_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                pred_texts.append(parts[1])

    # Ensure both lists have the same length
    if len(gt_texts) != len(pred_texts):
        logger.warning(
            f"Mismatch in number of samples: GT={len(gt_texts)}, Pred={len(pred_texts)}"
        )
        min_len = min(len(gt_texts), len(pred_texts))
        gt_texts = gt_texts[:min_len]
        pred_texts = pred_texts[:min_len]

    # Calculate WER
    if gt_texts and pred_texts:
        wer_score = calculate_wer_score(gt_texts, pred_texts)
        logger.info(f"WER Score for {test_data_name}: {wer_score:.4f}")
        return {test_data_name: wer_score}
    else:
        logger.error("No valid samples found for WER calculation")
        return {test_data_name: None}


def save_wer_results_to_consolidated_file(wer_results, adapter_dir, adapter_base):
    """Save WER results to a consolidated JSON file that contains all data cuts."""
    logger = logging.getLogger(__name__)
    consolidated_file = os.path.join(
        adapter_dir, f"{adapter_base}_all_wer_results.json"
    )

    # Load existing results if file exists
    existing_results = {}
    if os.path.exists(consolidated_file):
        try:
            with open(consolidated_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(
                f"Could not load existing results from {consolidated_file}, starting fresh"
            )
            existing_results = {}

    # Update with new results
    existing_results.update(wer_results)

    # Save consolidated results
    with open(consolidated_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

    logger.info(f"WER results saved to consolidated file: {consolidated_file}")
    return consolidated_file
