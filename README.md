# RL-LLM Training Pipeline

This repository contains a training pipeline for fine-tuning language models using SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), and RLOO (REINFORCE Leave One-Out) methods, with integrated Weights & Biands tracking and Nemotron reward model evaluation.

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

3. Set up Weights & Biands:
```bash
wandb login
```

## Running the Complete Pipeline

You can run the entire training and evaluation pipeline with a single script:

```bash
./run_all.sh
```

This script will:
1. Run SFT training on the SmolTalk dataset
2. Run DPO training (using the SFT model) on the UltraFeedback dataset
3. Run RLOO training on the UltraFeedback dataset
4. Evaluate all three models using the Nemotron 70B reward model
5. Log all results to Weights & Biands

The full pipeline will run in sequence and can take several hours to complete. Consider using tmux to run it in the background:

```bash
tmux new-session -d -s rl_llm_training './run_all.sh'
tmux attach-session -t rl_llm_training  # To monitor progress
```

## Memory Management

The pipeline includes automatic GPU memory management:
- GPU memory is cleared between training phases
- Memory is also cleared between batches and epochs
- Each training function uses a try/except/finally pattern to ensure proper cleanup

## Attention Mask Handling

The pipeline implements sophisticated attention mask handling for proper training:

1. **Prompt/Response Masks**:
   - Binary masks distinguishing between prompt tokens (0) and response tokens (1)
   - Created during dataset preparation based on message roles
   - Ensures that loss is only calculated on response tokens

2. **Padding Masks**:
   - Identifies padding tokens to exclude them from loss calculation
   - Works with standard padding token IDs (0, 1, -100)

3. **Combined Masking**:
   - The system combines attention masks, prompt/response masks, and padding masks
   - Results in loss being calculated only on valid response tokens
   - Improves training stability and performance

## Dataset Filtering

The pipeline automatically filters datasets to ensure high-quality training:

- Removes examples with excessively long prompts (>70% of the maximum sequence length)
- Preserves examples with adequate space for responses
- Reports statistics on filtered examples
- Enhances training stability by preventing truncation issues

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

For SmolTalk dataset, the pipeline automatically calculates an optimal max_length using the 95th percentile of token lengths from the dataset, so the specified max_length serves as an initial value.

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
    --sft_model_path "outputs/sft/final" \
    --dataset_name "ultrafeedback" \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --max_length 512 \
    --output_dir "outputs/rloo" \
    --use_wandb
```

## Evaluation

The evaluation pipeline uses the Nemotron 70B reward model (via NVIDIA API) to compare the trained model against a reference model.

To evaluate a trained model:
```bash
python evaluate.py \
    --model_path "outputs/dpo/final" \
    --nemotron_api_key "your_api_key" \
    --num_prompts 100 \
    --output_dir "outputs/dpo/eval" \
    --use_wandb
```

The pipeline uses the "test_prefs" split of the UltraFeedback dataset for evaluation, ensuring proper separation between training and testing data.

## Testing Pipeline

For quick testing or debugging, you can use the test pipeline:

```bash
./run_test.sh
```

This script runs a smaller version of the complete pipeline with limited data for faster testing.

## Numerical Stability

The training pipeline includes several measures to ensure numerical stability:

1. **Epsilon Values**: Small constants added to denominators to prevent division by zero
2. **Value Clamping**: Limits applied to log probabilities and advantage values to prevent overflow/underflow 
3. **NaN Detection**: Automatic detection and replacement of NaN values with fallback constants
4. **Gradient Clipping**: Applied to prevent exploding gradients
5. **Attention Mask Management**: Proper handling of masks for loss calculation

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
2. UltraFeedback (`HuggingFaceH4/ultrafeedback_binarized`): A dataset of preference pairs for DPO and RLOO training

For UltraFeedback, the pipeline uses:
- "train_prefs" split for training
- "test_prefs" split for evaluation

## Training Methods

1. **SFT (Supervised Fine-Tuning)**
   - Trains the model on high-quality conversations
   - Uses standard language modeling loss (on response tokens only)
   - Best for initial fine-tuning

2. **DPO (Direct Preference Optimization)**
   - Trains the model to prefer chosen responses over rejected ones
   - Uses preference pairs from UltraFeedback dataset
   - Requires a pre-trained SFT model
   - Formula: -E[log σ(β * (log πθ(yw|x)/πref(yw|x) - log πθ(yl|x)/πref(yl|x)))]

3. **RLOO (REINFORCE Leave One-Out)**
   - A policy gradient estimator based on REINFORCE with a variance-reducing baseline
   - The baseline for each sample is the weighted average of the rewards of other samples
   - Formally, the objective is:
     
     ![equation](https://latex.codecogs.com/svg.latex?\frac{1}{k}\sum_{i=1}^{k}\left[R(y^{(i)},x)-\frac{1}{k-1}\sum_{j\neq&space;i}R(y^{(j)},x)\right]\nabla\log&space;\pi(y^{(i)}|x))
     
     where y^(1),...,y^(k) are i.i.d samples from policy π_θ(·|x)
   - Generates multiple samples for each input to estimate the gradient

## Evaluation Metrics

The Nemotron reward model evaluation provides the following metrics:
- **Win rate**: Percentage of times the trained model's responses are preferred
- **Average model reward**: Mean reward score for trained model responses
- **Average reference reward**: Mean reward score for reference model responses
- **Reward improvement**: Difference between trained and reference rewards

The reward model uses the Bradley-Terry preference model to score responses, with higher scores indicating better quality.
