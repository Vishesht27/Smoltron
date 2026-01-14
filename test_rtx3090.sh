#!/bin/bash

# Quick Test Script for Nanotron on Single RTX 3090
# This script will run a minimal training test to verify the pipeline works

set -e  # Exit on error

echo "=================================="
echo "Nanotron RTX 3090 Test Script"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "run_train.py" ]; then
    echo "Error: run_train.py not found. Please run this script from the Smoltron directory."
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. CUDA might not be installed."
    exit 1
fi

nvidia-smi
echo ""

# Check if config exists
CONFIG_FILE="examples/config_test_rtx3090.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found."
    exit 1
fi

echo "Using config: $CONFIG_FILE"
echo ""

# Create checkpoints directory
mkdir -p checkpoints_test
echo "Created checkpoints directory: checkpoints_test"
echo ""

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=disabled  # Disable W&B for testing

echo "Environment variables set:"
echo "  CUDA_DEVICE_MAX_CONNECTIONS=1"
echo "  WANDB_MODE=disabled"
echo ""

# Run training
echo "=================================="
echo "Starting training test..."
echo "=================================="
echo ""

torchrun --nproc_per_node=1 run_train.py --config-file $CONFIG_FILE

echo ""
echo "=================================="
echo "Training test completed!"
echo "=================================="
echo ""

# Check if checkpoints were created
if [ -d "checkpoints_test" ]; then
    echo "Checkpoints created:"
    ls -lh checkpoints_test/
else
    echo "Warning: No checkpoints directory found."
fi

echo ""
echo "Test completed successfully! ✓"
