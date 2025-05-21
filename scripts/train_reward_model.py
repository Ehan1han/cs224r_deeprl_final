import os
import argparse
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from rl_llm.data import PreferenceDataset, create_dataloader
from rl_llm.reward import RewardModel, RewardTrainer

def train_reward_model(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    dataset_name: str = "ultrafeedback",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 5,
    max_length: int = 512,
    hidden_size: int = 1024,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    output_dir: str = "outputs/reward_model",
    use_wandb: bool = True,
    gradient_accumulation_steps: int = 4,
    eval_steps: int = 500,
    save_steps: int = 1000,
    max_grad_norm: float = 1.0,
    subset_size: int = None
):
    """Train reward model using Bradley-Terry objective."""
    try:
        # Initialize wandb if enabled
        if use_wandb:
            run = wandb.init(
                entity="zhao111han-stanford-university",
                project="cs224r_deeprl_final",
                name=f"reward_{model_name.split('/')[-1]}_{dataset_name}",
                config={
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "max_length": max_length,
                    "hidden_size": hidden_size,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "method": "reward_model",
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_batch_size": batch_size * gradient_accumulation_steps,
                    "subset_size": subset_size
                }
            )
        
        # Initialize model and tokenizer
        model = RewardModel(
            model_name=model_name,
            hidden_size=hidden_size,
            dropout=dropout
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create datasets
        train_dataset = PreferenceDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split="train_prefs",
            max_length=max_length,
            subset_size=subset_size
        )
        
        val_dataset = PreferenceDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split="test_prefs",
            max_length=max_length,
            subset_size=subset_size
        )
        
        train_dataloader = create_dataloader(train_dataset, batch_size=batch_size)
        val_dataloader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Print dataset information
        print(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)} examples")
        
        # Initialize trainer with learning rate schedule
        trainer = RewardTrainer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training loop
        global_step = 0
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_accuracy = 0
            total_steps = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Training step
                metrics = trainer.train_step(
                    chosen_ids=batch["chosen_input_ids"],
                    rejected_ids=batch["rejected_input_ids"],
                    chosen_mask=batch["chosen_attention_mask"],
                    rejected_mask=batch["rejected_attention_mask"]
                )
                
                total_loss += metrics["loss"]
                total_accuracy += metrics["accuracy"]
                total_steps += 1
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "lr": trainer.get_current_lr()
                })
                
                # Log metrics to wandb
                if use_wandb:
                    wandb.log({
                        "train/loss": metrics["loss"],
                        "train/accuracy": metrics["accuracy"],
                        "train/chosen_rewards": metrics["chosen_rewards"],
                        "train/rejected_rewards": metrics["rejected_rewards"],
                        "train/learning_rate": trainer.get_current_lr()
                    })
                
                # Evaluation
                if global_step % eval_steps == 0:
                    val_metrics = evaluate(model, val_dataloader, device)
                    print(f"\nStep {global_step} Validation:", val_metrics)
                    
                    if use_wandb:
                        wandb.log({
                            "val/loss": val_metrics["loss"],
                            "val/accuracy": val_metrics["accuracy"],
                            "val/chosen_rewards": val_metrics["chosen_rewards"],
                            "val/rejected_rewards": val_metrics["rejected_rewards"]
                        })
                    
                    # Save best model
                    if val_metrics["accuracy"] > best_val_accuracy:
                        best_val_accuracy = val_metrics["accuracy"]
                        os.makedirs(output_dir, exist_ok=True)
                        model.save_pretrained(os.path.join(output_dir, "best"))
                        print(f"Saved new best model with accuracy: {best_val_accuracy:.4f}")
                
                # Regular checkpoint saving
                if global_step % save_steps == 0:
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(
                        os.path.join(output_dir, f"checkpoint-{global_step}")
                    )
            
            # Compute epoch metrics
            avg_loss = total_loss / total_steps
            avg_accuracy = total_accuracy / total_steps
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
            
            # Log epoch metrics to wandb
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/avg_loss": avg_loss,
                    "epoch/avg_accuracy": avg_accuracy
                })
        
        # Save final model
        model.save_pretrained(os.path.join(output_dir, "final"))
        if use_wandb:
            run.finish()
            
    except Exception as e:
        print(f"Reward model training failed: {str(e)}")
        if use_wandb and 'run' in locals():
            run.finish()
        raise e

def evaluate(model, dataloader, device):
    """Evaluate the reward model on validation data."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_chosen_rewards = 0
    total_rejected_rewards = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, metrics = model.compute_loss(
                chosen_ids=batch["chosen_input_ids"],
                rejected_ids=batch["rejected_input_ids"],
                chosen_mask=batch["chosen_attention_mask"],
                rejected_mask=batch["rejected_attention_mask"]
            )
            
            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            total_chosen_rewards += metrics["chosen_rewards"]
            total_rejected_rewards += metrics["rejected_rewards"]
            total_steps += 1
    
    return {
        "loss": total_loss / total_steps,
        "accuracy": total_accuracy / total_steps,
        "chosen_rewards": total_chosen_rewards / total_steps,
        "rejected_rewards": total_rejected_rewards / total_steps
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="ultrafeedback")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="outputs/reward_model")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--subset_size", type=int, default=None)
    args = parser.parse_args()
    
    train_reward_model(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        subset_size=args.subset_size
    )
