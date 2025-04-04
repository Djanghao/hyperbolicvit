#!/bin/bash

# Default dataset is CIFAR-10, but can be changed to 'imagenet'
DATASET=${1:-"cifar10"}
# Default subset ratio is 0.05 (5% of the dataset)
SUBSET_RATIO=${2:-"0.05"}

# Activate the virtual environment
source hypervit_env/bin/activate

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Set environment variables
export DATASET=$DATASET
export SUBSET_RATIO=$SUBSET_RATIO

# Print the training configuration
echo "Starting HyperbolicViT training (single GPU) with dataset: $DATASET (using ${SUBSET_RATIO} of data)"

# Create output directory if it doesn't exist
mkdir -p ./output

# Run the training script
python train_single_gpu.py
