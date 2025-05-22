#!/bin/bash
set -e

# Activate environment with more explicit approach
echo "Activating conda environment..."
export PATH=~/miniconda3/bin:$PATH
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate rl_llm

# Explicitly check Python environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $(conda info --envs | grep '*')"
echo "Checking for transformers: $(python -c "import transformers; print(f'transformers {transformers.__version__} found')" 2>&1 || echo "transformers not found")"

wandb login


# Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true

# Train reward model on subset (100 examples) for faster experiments
echo "Training reward model on 100 examples subset..."
python train_reward_model.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "ultrafeedback" \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --num_epochs 5 \
    --max_length 1024 \
    --output_dir "outputs/reward_model_100" \
    --subset_size 100 \
    --use_wandb


# Clear GPU memory before starting
echo "Clearing GPU memory before starting..."
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true

# Train reward model on full dataset
echo "Training reward model on full UltraFeedback dataset..."
python train_reward_model.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "ultrafeedback" \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --num_epochs 1 \
    --max_length 1024 \
    --output_dir "outputs/reward_model_full" \
    --use_wandb

echo "Reward models training completed!" 