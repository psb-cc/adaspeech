"""
Usage:
python infer.py \
    --path_to_config_file ./configs/openai-whisper-large-v3_multi-res-conv-small-projector_krutrim-2-12b-instruct.yaml \
    --path_to_adapter_weights ./checkpoints/hindi__openai-whisper-large-v3_multi-res-conv-small-projector_krutrim-2-12b-instruct/best_model_no_loss.pt \
    --path_to_test_manifest_file ./data/hindi_noisy_test_known.jsonl
"""

import argparse
import logging
import os
import time

import torch
import yaml
from tqdm import tqdm

from src.dataset import AsrDataset
from src.model import SpeechEncoder2Adapter2Llm
from src.utils import (
    batch_to_device,
    read_and_calculate_wer,
    save_rtfx_score_to_consolidated_file,
    save_wer_results_to_consolidated_file,
    set_random_seed,
)


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with adapter")
    parser.add_argument(
        "--path_to_config_file",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--path_to_adapter_weights",
        type=str,
        required=True,
        help="Path to adapter weights",
    )
    parser.add_argument(
        "--path_to_test_manifest_file",
        type=str,
        required=True,
        help="Path to input jsonl file for inference",
    )
    parser.add_argument(
        "--path_to_lora_adapter_weights",
        type=str,
        required=False,
        help="Path to lora adapter weights",
    )
    return parser.parse_args()


args = parse_args()
print(args)
config = load_config(args.path_to_config_file)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

for key, value in config["env"].items():
    os.environ[key.upper()] = str(value)

set_random_seed()

adapter_dir = os.path.dirname(args.path_to_adapter_weights)
adapter_base = os.path.splitext(os.path.basename(args.path_to_adapter_weights))[0]

test_data_name = os.path.splitext(os.path.basename(args.path_to_test_manifest_file))[0]
path_to_save_gt_file = os.path.join(adapter_dir, f"{adapter_base}_{test_data_name}_gt")
path_to_save_pred_file = os.path.join(
    adapter_dir, f"{adapter_base}_{test_data_name}_pred"
)

log_file_path = os.path.join(adapter_dir, f"{adapter_base}_inference.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

lora_cfg = config["training"].get("lora_config", None)
model = SpeechEncoder2Adapter2Llm(
    config["model"]["speech_encoder_name"],
    config["model"]["llm_name"],
    config["model"]["adapter_name"],
    device="cuda:0",
    path_to_adapter_weights=args.path_to_adapter_weights,
    lora_config_dict=lora_cfg,
    lora_adapter_ckpt_path=args.path_to_lora_adapter_weights,
)

test_dataset = AsrDataset(
    tokenizer=model.tokenizer,
    prompt=config["model"]["prompt"],
    mel_size=config["model"]["mel_size"],
    fix_length_audio=config["model"]["fix_length_audio"],
    fix_audio_duration=config["model"]["fix_audio_duration"],
    inference_mode=True,
    normalize=config["model"]["normalize"],
    input_type=config["model"]["input_type"],
    path_to_jsonl_file=args.path_to_test_manifest_file,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config["inference"]["batch_size"],
    shuffle=False,
    pin_memory=True,
    collate_fn=test_dataset.collator,
    drop_last=False,
    num_workers=config["inference"]["num_workers"],
)

logger.info(
    f"Model: {model.__class__.__name__} with {model.speech_encoder.__class__.__name__} -> {model.adapter.__class__.__name__} -> {model.llm.__class__.__name__}"
)
logger.info("Starting inference with configuration:")
logger.info(f"Adapter weights path: {args.path_to_adapter_weights}")
logger.info(f"Test manifest file: {args.path_to_test_manifest_file}")
logger.info(f"Batch size: {config['inference']['batch_size']}")
logger.info(f"Number of workers: {config['inference']['num_workers']}")
logger.info(f"Total samples: {len(test_dataset)}")
logger.info(f"Total batches: {len(test_dataloader)}")
logger.info(f"Ground truth file: {path_to_save_gt_file}")
logger.info(f"Predictions file: {path_to_save_pred_file}")
logger.info(f"Log file: {log_file_path}")

model.eval()
model.llm.model.generation_config.pad_token_id = model.tokenizer.pad_token_id

all_audio_durations = []
all_transcription_times = []

with open(path_to_save_gt_file, "w") as gt, open(path_to_save_pred_file, "w") as pred:
    progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for step, batch in progress_bar:

        # rtfx code adapted from: https://github.com/huggingface/open_asr_leaderboard/blob/main/transformers/run_eval.py
        start_time = time.time()  # NOTE: for rtfx
        batch = batch_to_device(batch, "cuda:0")
        model_outputs = model.generate(batch)
        output_text = model.tokenizer.batch_decode(
            model_outputs, add_special_tokens=False, skip_special_tokens=True
        )
        runtime = time.time() - start_time  # NOTE: for rtfx

        minibatch_size = len(batch["audio_duration"])
        per_sample_runtime = runtime / minibatch_size
        all_transcription_times.extend([per_sample_runtime] * minibatch_size)
        all_audio_durations.extend(batch["audio_duration"].cpu().numpy().tolist())

        current_rtfx = sum(all_audio_durations) / sum(all_transcription_times)

        for key, text, target in zip(batch["keys"], output_text, batch["targets"]):
            pred.write(key + "\t" + text.replace("\n", " ") + "\n")
            gt.write(key + "\t" + target + "\n")
            if step % 5 == 0:
                cleaned_text = text.replace("\n", " ")
                progress_bar.set_description(
                    f"[Step: {step}]|[Pred: {cleaned_text} | GT: {target}]|[RTFX: {current_rtfx:.2f}]"
                )

final_rtfx = {test_data_name: sum(all_audio_durations) / sum(all_transcription_times)}
logger.info(f"Final RTFx Score: {final_rtfx[test_data_name]:.2f}")

consolidated_rtfx_file = save_rtfx_score_to_consolidated_file(
    final_rtfx, adapter_dir, adapter_base
)


logger.info("Calculating WER score...")
wer_results = read_and_calculate_wer(
    path_to_save_gt_file, path_to_save_pred_file, test_data_name
)

consolidated_wer_file = save_wer_results_to_consolidated_file(
    wer_results, adapter_dir, adapter_base
)

logger.info("Inference completed successfully!")
logger.info(f"Results saved to:")
logger.info(f"  Ground truth: {path_to_save_gt_file}")
logger.info(f"  Predictions: {path_to_save_pred_file}")
logger.info(f"  WER results: {consolidated_wer_file}")
logger.info(f"  WER Score: {wer_results.get(test_data_name, 'N/A')}")
