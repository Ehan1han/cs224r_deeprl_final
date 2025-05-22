import os
import argparse
import torch
import wandb
import gc
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Optional

from rl_llm.data import PreferenceDataset, create_dataloader
from rl_llm.reward import RewardModel, RewardTrainer

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def train_reward_model(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    dataset_name: str = "ultrafeedback",
    batch_size: int = 8,
    learning_rate: float = 5e-6,
    num_epochs: int = 3,
    max_length: int = 1024,
    output_dir: str = "outputs/reward_model",
    use_wandb: bool = True,
    subset_size: Optional[int] = None,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1
):
    """Train a reward model using Bradley-Terry preference learning."""
    try:
        # Initialize wandb if enabled
        if use_wandb:
            run = wandb.init(
                entity="zhao111han-stanford-university",
                project="cs224r_deeprl_final",
                name=f"reward_model_{model_name.split('/')[-1]}_{dataset_name}{'_'+str(subset_size) if subset_size else ''}",
                config={
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "max_length": max_length,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "subset_size": subset_size
                }
            )
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create dataset and dataloader
        dataset = PreferenceDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=max_length,
            subset_size=subset_size
        )
        dataloader = create_dataloader(dataset, batch_size=batch_size)
        
        # Print dataset information
        print(f"Training reward model on {len(dataset)} examples from {dataset_name}" + 
              (f" (subset: {subset_size})" if subset_size else ""))
        
        # Initialize reward model with LoRA adaptation
        reward_model = RewardModel(
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        print(f"Initialized LoRA-adapted reward model based on {model_name}")
        
        # Initialize reward trainer
        reward_trainer = RewardTrainer(
            model=reward_model,
            learning_rate=learning_rate,
            weight_decay=0.01
        )
        
        # Training reward model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_steps = 0
        
        for epoch in range(num_epochs):
            print(f"Reward model training epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_steps = 0
            
            for batch in tqdm(dataloader, desc=f"Reward Model Epoch {epoch+1}"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Train step
                metrics = reward_trainer.train_step(
                    chosen_ids=batch["chosen_input_ids"],
                    rejected_ids=batch["rejected_input_ids"],
                    chosen_mask=batch["chosen_attention_mask"],
                    rejected_mask=batch["rejected_attention_mask"]
                )
                
                epoch_loss += metrics["loss"]
                epoch_accuracy += metrics["accuracy"]
                epoch_steps += 1
                total_steps += 1
                
                # Log metrics
                if use_wandb:
                    wandb.log({
                        "reward_model/loss": metrics["loss"],
                        "reward_model/accuracy": metrics["accuracy"],
                        "reward_model/chosen_rewards": metrics["chosen_rewards"],
                        "reward_model/rejected_rewards": metrics["rejected_rewards"],
                        "reward_model/step": total_steps
                    })
            
            # Log epoch metrics
            avg_loss = epoch_loss / epoch_steps
            avg_accuracy = epoch_accuracy / epoch_steps
            print(f"Reward model epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            
            if use_wandb:
                wandb.log({
                    "reward_model/epoch_loss": avg_loss,
                    "reward_model/epoch_accuracy": avg_accuracy,
                    "reward_model/epoch": epoch+1
                })
        
        # Save the trained reward model
        os.makedirs(output_dir, exist_ok=True)
        reward_model.save_pretrained(output_dir)
        print(f"Reward model saved to {output_dir}")
        
        if use_wandb:
            run.finish()
            
    except Exception as e:
        print(f"Reward model training failed: {str(e)}")
        if use_wandb and 'run' in locals():
            run.finish()
        raise e
    finally:
        clear_gpu_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model for RLOO")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model to use for reward model")
    parser.add_argument("--dataset_name", type=str, default="ultrafeedback", help="Dataset to train on")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="outputs/reward_model", help="Directory to save the reward model")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use from dataset (for debugging/testing)")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank of LoRA adaptation matrices")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")
    
    args = parser.parse_args()
    
    train_reward_model(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        subset_size=args.subset_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    ) 