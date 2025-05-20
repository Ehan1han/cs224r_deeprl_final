#!/bin/bash
set -e

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_llm

# Login to wandb - no need to export API key since we already logged in
wandb login

# Clear GPU memory before starting
echo "Clearing GPU memory before starting..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true

# SFT
echo "Training SFT model..."
python train.py --method sft --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4 --learning_rate 1e-6 --num_epochs 1 --max_length 512 --output_dir "outputs/sft" --use_wandb --gradient_accumulation_steps 4

# Verify SFT model exists
if [ ! -d "outputs/sft/final" ]; then
    echo "Error: SFT model not found at outputs/sft/final"
    exit 1
fi

# Clear GPU memory after SFT
echo "Clearing GPU memory after SFT..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true

# DPO (using fine-tuned SFT model)
echo "Training DPO model..."
python train.py --method dpo --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4 --learning_rate 1e-6 --num_epochs 1 --max_length 512 --output_dir "outputs/dpo" --sft_model_path "outputs/sft/final" --use_wandb --gradient_accumulation_steps 4

# Clear GPU memory after DPO
echo "Clearing GPU memory after DPO..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true

# RLOO
echo "Training RLOO model..."
python train.py --method rloo --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4 --learning_rate 1e-6 --num_epochs 1 --max_length 512 --output_dir "outputs/rloo" --use_wandb --gradient_accumulation_steps 4

# Clear GPU memory after RLOO
echo "Clearing GPU memory after RLOO..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true

# Evaluate all models
echo "Evaluating SFT model..."
python evaluate.py --model_path "outputs/sft/final" --num_prompts 100 --output_dir "outputs/sft/eval" --use_wandb

echo "Evaluating DPO model..."
python evaluate.py --model_path "outputs/dpo/final" --num_prompts 100 --output_dir "outputs/dpo/eval" --use_wandb

echo "Evaluating RLOO model..."
python evaluate.py --model_path "outputs/rloo/final" --num_prompts 100 --output_dir "outputs/rloo/eval" --use_wandb

echo "All done!" 