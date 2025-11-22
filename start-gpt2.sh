#!/bin/bash

# ==============================
# Run file for GPT2 example - how to use the launch script to run GPT2 with simple trainer
# ==============================

### run GPT2 Simple Trainer
export CUDA_VISIBLE_DEVICES=5
python launch.py --model gpt2 --trainer simple --batch_size 4 --dataset_split "train[:100]" --train_stats simple "$@"
