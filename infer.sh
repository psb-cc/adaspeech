#!/bin/bash

# Script to run inference with the best model checkpoint
# Usage: ./infer.sh <path_to_config_file>

set -e  # Exit on any error

# Check if correct number of arguments provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_config_file>"
    echo "Example: $0 configs/MMS_linear_llama.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

echo "=== Speech-to-Text Inference Script ==="
echo "Config file: $CONFIG_FILE"
echo

# Extract experiment name and data directory from config file
EXPERIMENT_NAME=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config['wandb']['name'])
except Exception as e:
    print(f'Error reading config: {e}', file=sys.stderr)
    sys.exit(1)
")

DATA_DIR=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config['paths']['data_dir'])
except Exception as e:
    print(f'Error reading config: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: Could not extract experiment name from config file!"
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    echo "Error: Could not extract data directory from config file!"
    exit 1
fi

echo "Experiment name: $EXPERIMENT_NAME"
echo "Data directory: $DATA_DIR"

# Extract list of test manifest files
TEST_MANIFESTS=$(python3 -c "
import yaml
import sys
import os
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    data_dir = config['paths']['data_dir']
    manifest_files = config['paths']['list_of_test_manifest_files']
    for manifest in manifest_files:
        full_path = os.path.join(data_dir, manifest)
        print(full_path)
except Exception as e:
    print(f'Error reading config: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ -z "$TEST_MANIFESTS" ]; then
    echo "Error: Could not extract test manifest files from config file!"
    exit 1
fi

echo "Test manifest files to process:"
echo "$TEST_MANIFESTS"
echo

# Construct path to checkpoints directory
CHECKPOINTS_DIR="checkpoints/$EXPERIMENT_NAME"

# Check if checkpoints directory exists
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Error: Checkpoints directory '$CHECKPOINTS_DIR' not found!"
    echo "Available checkpoint directories:"
    ls -la checkpoints/ 2>/dev/null || echo "No checkpoints directory found"
    exit 1
fi

# Look for the best model checkpoint (prefer best_model_no_loss.pt, fallback to best_model_val_loss.pt)
BEST_MODEL_PATH=""
if [ -f "$CHECKPOINTS_DIR/best_model_no_loss.pt" ]; then
    BEST_MODEL_PATH="$CHECKPOINTS_DIR/best_model_no_loss.pt"
    echo "Found best model (no loss): $BEST_MODEL_PATH"
elif [ -f "$CHECKPOINTS_DIR/best_model_val_loss.pt" ]; then
    BEST_MODEL_PATH="$CHECKPOINTS_DIR/best_model_val_loss.pt"
    echo "Found best model (val loss): $BEST_MODEL_PATH"
else
    echo "Error: No base adapter model checkpoint found in '$CHECKPOINTS_DIR'!"
    echo "Looking for: best_model_no_loss.pt or best_model_val_loss.pt"
    echo "Available files:"
    ls -la "$CHECKPOINTS_DIR"/*.pt 2>/dev/null || echo "No .pt files found"
    exit 1
fi

# Look for any LoRA adapter checkpoint (optional)
LORA_CKPT=""
if [ -f "$CHECKPOINTS_DIR/lora_adapter_checkpoint.pt" ]; then
    LORA_CKPT="$CHECKPOINTS_DIR/lora_adapter_checkpoint.pt"
    echo "Found LoRA adapter checkpoint: $LORA_CKPT"
else
    echo "No LoRA adapter checkpoint found. Proceeding without it."
fi

echo "Using base adapter model checkpoint: $BEST_MODEL_PATH"
if [ -n "$LORA_CKPT" ]; then
    echo "Using LoRA adapter checkpoint: $LORA_CKPT"
fi
echo

# Run inference for each test manifest file
echo "=== Starting Inference ==="
while IFS= read -r TEST_MANIFEST; do
    echo "Processing: $TEST_MANIFEST"
    
    # Check if test manifest file exists
    if [ ! -f "$TEST_MANIFEST" ]; then
        echo "Warning: Test manifest file '$TEST_MANIFEST' not found! Skipping..."
        continue
    fi
    
    echo "Running inference on: $TEST_MANIFEST"
    python3 infer.py \
        --path_to_config_file "$CONFIG_FILE" \
        --path_to_adapter_weights "$BEST_MODEL_PATH" \
        --path_to_test_manifest_file "$TEST_MANIFEST" \
        --path_to_lora_adapter_weights "$LORA_CKPT"
    
    echo "Completed processing: $TEST_MANIFEST"
    echo "---"
done <<< "$TEST_MANIFESTS"

echo
echo "=== Inference Complete ==="
echo "Results saved in: $CHECKPOINTS_DIR"
