#!/bin/bash
set -e

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_llm

# Login to wandb
export WANDB_API_KEY="e654d6407fe0da95379d8938659cdb1ddf7fb857"
wandb login

# SFT
python train.py --method sft --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4  --learning_rate 1e-5 --num_epochs 3 --max_length 256 --output_dir "outputs/sft"

# Verify SFT model exists
if [ ! -d "outputs/sft/final" ]; then
    echo "Error: SFT model not found at outputs/sft/final"
    exit 1
fi

# DPO (using fine-tuned SFT model)
python train.py --method dpo --model_name "Qwen/Qwen2.5-0.5B" --batch_size 2 --learning_rate 1e-5 --num_epochs 3 --max_length 256 --output_dir "outputs/dpo" --sft_model_path "outputs/sft/final"

# RLOO
python train.py --method rloo --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4 --learning_rate 1e-5 --num_epochs 3 --max_length 256 --output_dir "outputs/rloo"

# Evaluate SFT
python train.py --method eval --model_path outputs/sft/final --num_prompts 100 --output_dir outputs/sft/eval

# Evaluate DPO
python train.py --method eval --model_path outputs/dpo/final --num_prompts 100 --output_dir outputs/dpo/eval

# Evaluate RLOO
python train.py --method eval --model_path outputs/rloo/final --num_prompts 100 --output_dir outputs/rloo/eval

echo "All done!" 