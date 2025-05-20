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

echo "Evaluating SFT model..."
python evaluate.py --model_path "outputs/sft/final" --num_prompts 100 --output_dir "outputs/sft/eval" --use_wandb

# Clear GPU memory after SFT
echo "Clearing GPU memory after SFT..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true



# DPO (using fine-tuned SFT model)
echo "Training DPO model with 100 examples subset..."
python train.py --method dpo --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4 --learning_rate 2e-6 --num_epochs 1 --max_length 512 --output_dir "outputs/dpo_100" --sft_model_path "outputs/sft/final" --use_wandb --gradient_accumulation_steps 8 --subset_size 100 --max_steps 500


echo "Evaluating DPO model with 100 examples..."
python evaluate.py --model_path "outputs/dpo_100/final" --num_prompts 100 --output_dir "outputs/dpo_100/eval" --use_wandb

# Clear GPU memory after DPO 100
echo "Clearing GPU memory after DPO 100 examples training..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true



# DPO with 1000 examples
echo "Training DPO model with 1000 examples subset..."
python train.py --method dpo --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4 --learning_rate 2e-6 --num_epochs 1 --max_length 512 --output_dir "outputs/dpo_1000" --sft_model_path "outputs/sft/final" --use_wandb --gradient_accumulation_steps 8 --subset_size 1000 --max_steps 2000

echo "Evaluating DPO model with 1000 examples..."
python evaluate.py --model_path "outputs/dpo_1000/final" --num_prompts 100 --output_dir "outputs/dpo_1000/eval" --use_wandb

# Clear GPU memory after DPO
echo "Clearing GPU memory after DPO 1000 examples training..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true


# RLOO
# echo "Training RLOO model..."
# python train.py --method rloo --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4 --learning_rate 1e-6 --num_epochs 1 --max_length 512 --output_dir "outputs/rloo" --sft_model_path "outputs/sft/final" --use_wandb --gradient_accumulation_steps 4

# Clear GPU memory after RLOO
# echo "Clearing GPU memory after RLOO..."
# python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true

# RLOO with 100 examples
echo "Training RLOO model with 100 examples subset..."
python train.py --method rloo --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4 --learning_rate 1e-6 --num_epochs 1 --max_length 512 --output_dir "outputs/rloo_100" --sft_model_path "outputs/sft/final" --use_wandb --gradient_accumulation_steps 4 --subset_size 100 --max_steps 500

echo "Evaluating RLOO model with 100 examples..."
python evaluate.py --model_path "outputs/rloo_100/final" --num_prompts 100 --output_dir "outputs/rloo_100/eval" --use_wandb

# Clear GPU memory after RLOO 100
echo "Clearing GPU memory after RLOO 100 examples training..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true




# RLOO with 1000 examples
echo "Training RLOO model with 1000 examples subset..."
python train.py --method rloo --model_name "Qwen/Qwen2.5-0.5B" --batch_size 4 --learning_rate 1e-6 --num_epochs 1 --max_length 512 --output_dir "outputs/rloo_1000" --sft_model_path "outputs/sft/final" --use_wandb --gradient_accumulation_steps 4 --subset_size 1000 --max_steps 2000

echo "Evaluating RLOO model with 1000 examples..."
python evaluate.py --model_path "outputs/rloo_1000/final" --num_prompts 100 --output_dir "outputs/rloo_1000/eval" --use_wandb

# Clear GPU memory after RLOO 1000
echo "Clearing GPU memory after RLOO 1000 examples training..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true


echo "All done!" 