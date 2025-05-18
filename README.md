# RL-LLM Training Pipeline

This repository contains a training pipeline for fine-tuning language models using SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), and RLOO (Reinforcement Learning with Offline Optimization) methods.

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

## Training

### SFT Training
```bash
python train.py --method sft \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "smoltalk" \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --max_length 512 \
    --output_dir "outputs/sft"
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
    --output_dir "outputs/dpo"
```

### RLOO Training
```bash
python train.py --method rloo \
    --sft_model_path "outputs/sft/final" \
    --dataset_name "smoltalk" \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --max_length 512 \
    --output_dir "outputs/rloo"
```

## Evaluation

To evaluate a trained model:
```bash
python evaluate.py \
    --model_path "outputs/dpo/final" \
    --nemotron_api_key "your_api_key" \
    --num_prompts 100
```

## Datasets

The pipeline supports two datasets:
1. SmolTalk (`HuggingFaceTB/smol-smoltalk`): A dataset of conversations for SFT and RLOO training
2. UltraFeedback (`HuggingFaceH4/ultrafeedback_binarized`): A dataset of preference pairs for DPO training

## Training Methods

1. **SFT (Supervised Fine-Tuning)**
   - Trains the model on high-quality conversations
   - Uses standard language modeling loss
   - Best for initial fine-tuning

2. **DPO (Direct Preference Optimization)**
   - Trains the model to prefer chosen responses over rejected ones
   - Uses preference pairs from UltraFeedback dataset
   - Requires a pre-trained SFT model

3. **RLOO (Reinforcement Learning with Offline Optimization)**
   - Uses offline reinforcement learning to optimize responses
   - Requires a reward model and pre-trained SFT model
   - Generates multiple samples for each input

## Evaluation

The evaluation pipeline uses the Nemotron reward model to compare the trained model against a reference model. Metrics include:
- Win rate: Percentage of times the trained model's responses are preferred
- Average reward: Mean reward score for trained model responses
- Average reference reward: Mean reward score for reference model responses
- Reward improvement: Difference between trained and reference rewards
