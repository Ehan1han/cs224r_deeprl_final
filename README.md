# RL-LLM Training Pipeline

This repository contains a training pipeline for fine-tuning language models using SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), and RLOO (REINFORCE Leave One-Out) methods, with integrated Weights & Biands tracking.

## Setup

1. Create a conda environment:
```bash
conda create -n rl_llm python=3.10
conda activate rl_llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases:
```bash
wandb login
```

## Running the Complete Pipeline

You can run the entire training and evaluation pipeline with a single script:

```bash
./run_all.sh
```

This script will:
1. Run SFT training
2. Run DPO training (using the SFT model)
3. Run RLOO training
4. Evaluate all three models
5. Log all results to Weights & Biands

The full pipeline will run in sequence and can take several hours to complete. Consider using tmux to run it in the background:

```bash
tmux new-session -d -s rl_llm_training './run_all.sh'
tmux attach-session -t rl_llm_training  # To monitor progress
```

## Individual Training Commands

### SFT Training
```bash
python train.py --method sft \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "smoltalk" \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --max_length 512 \
    --output_dir "outputs/sft" \
    --use_wandb
```

### DPO Training
```bash
python train.py --method dpo \
    --sft_model_path "outputs/sft/final" \
    --dataset_name "ultrafeedback" \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --max_length 512 \
    --output_dir "outputs/dpo" \
    --use_wandb
```

### RLOO Training
```bash
python train.py --method rloo \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "ultrafeedback" \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --max_length 512 \
    --output_dir "outputs/rloo" \
    --use_wandb
```

## Evaluation

To evaluate a trained model:
```bash
python evaluate.py \
    --model_path "outputs/dpo/final" \
    --nemotron_api_key "your_api_key" \
    --num_prompts 100 \
    --output_dir "outputs/dpo/eval" \
    --use_wandb
```

## Weights & Biands Integration

All training and evaluation runs are tracked with Weights & Biands. The integration includes:

- Training hyperparameters tracking
- Loss curves for each training method
- Evaluation metrics comparison
- Model checkpoints
- Run naming for easy identification

To view your runs, visit: https://wandb.ai/[your-username]/cs224r_deeprl_final

## Datasets

The pipeline supports two datasets:
1. SmolTalk (`HuggingFaceTB/smol-smoltalk`): A dataset of conversations for SFT 
2. UltraFeedback (`HuggingFaceH4/ultrafeedback_binarized`): A dataset of preference pairs for DPO training and RLOO training

## Training Methods

1. **SFT (Supervised Fine-Tuning)**
   - Trains the model on high-quality conversations
   - Uses standard language modeling loss
   - Best for initial fine-tuning

2. **DPO (Direct Preference Optimization)**
   - Trains the model to prefer chosen responses over rejected ones
   - Uses preference pairs from UltraFeedback dataset
   - Requires a pre-trained SFT model

3. **RLOO (REINFORCE Leave One-Out)**
   - A policy gradient estimator based on REINFORCE with a variance-reducing baseline
   - The baseline for each sample is the weighted average of the rewards of other samples
   - Formally, the objective is:
     
     ![equation](https://latex.codecogs.com/svg.latex?\frac{1}{k}\sum_{i=1}^{k}\left[R(y^{(i)},x)-\frac{1}{k-1}\sum_{j\neq&space;i}R(y^{(j)},x)\right]\nabla\log&space;\pi(y^{(i)}|x))
     
     where y^(1),...,y^(k) are i.i.d samples from policy π_θ(·|x)
   - Generates multiple samples for each input to estimate the gradient

## Evaluation

The evaluation pipeline uses the Nemotron reward model to compare the trained model against a reference model. Metrics include:
- Win rate: Percentage of times the trained model's responses are preferred
- Average reward: Mean reward score for trained model responses
- Average reference reward: Mean reward score for reference model responses
- Reward improvement: Difference between trained and reference rewards
- Length ratio: Ratio of trained model response length to reference model response length
