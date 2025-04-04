#!/bin/bash

# Default dataset is CIFAR-10, but can be changed to 'imagenet'
DATASET=${1:-"cifar10"}

# Activate the virtual environment
source hypervit_env/bin/activate

# Set the GPU to use (adjust according to your available GPUs)
export CUDA_VISIBLE_DEVICES=0

# Set the dataset environment variable
export DATASET=$DATASET

# Print the training configuration
echo "Starting HyperbolicViT training with dataset: $DATASET"

# Run the training script
python ddp.py
