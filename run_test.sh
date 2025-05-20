#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_llm

# Run the test pipeline
python test_pipeline.py

# Evaluate all three models
echo "Evaluating SFT model..."
python evaluate.py \
    --model_path "outputs/test/sft/final" \
    --num_prompts 10 \
    --output_dir "outputs/test/sft/eval" \
    --use_wandb

echo "Evaluating DPO model..."
python evaluate.py \
    --model_path "outputs/test/dpo/final" \
    --num_prompts 10 \
    --output_dir "outputs/test/dpo/eval" \
    --use_wandb

echo "Evaluating RLOO model..."
python evaluate.py \
    --model_path "outputs/test/rloo/final" \
    --num_prompts 10 \
    --output_dir "outputs/test/rloo/eval" \
    --use_wandb 