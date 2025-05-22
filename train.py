import os
import argparse
import torch
import wandb
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
import json
from typing import Optional

# Set multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

from rl_llm.data import PreferenceDataset, create_dataloader
from rl_llm.model_utils import QwenModel
from rl_llm.objectives import SFTTrainer, DPOTrainer, RLOOTrainer
from rl_llm.reward import RewardModel, RewardTrainer
from rl_llm.evaluation import run_evaluation

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def train_sft(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    dataset_name: str = "smoltalk",
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    output_dir: str = "outputs/sft",
    use_wandb: bool = True,
    dataloader: Optional[DataLoader] = None,
    gradient_accumulation_steps: int = 4
):
    """Train model using SFT."""
    try:
        # Initialize wandb if enabled
        if use_wandb:
            run = wandb.init(
                entity="zhao111han-stanford-university",
                project="cs224r_deeprl_final",
                name=f"sft_{model_name.split('/')[-1]}_{dataset_name}",
                config={
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "max_length": max_length,
                    "method": "sft",
                    "effective_batch_size": batch_size * gradient_accumulation_steps
                }
            )
        
        # Initialize model and tokenizer
        model = QwenModel(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create dataset and dataloader if not provided
        if dataloader is None:
            dataset = PreferenceDataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=max_length
            )
            # If using SmolTalk, max_length may have been adjusted based on 95th percentile
            if dataset_name == "smoltalk" and hasattr(dataset, "max_length"):
                max_length = dataset.max_length
                print(f"Using calculated max_length from dataset: {max_length}")
                
            dataloader = create_dataloader(dataset, batch_size=batch_size)
        
        # Initialize trainer
        trainer = SFTTrainer(
            model, 
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            try:
                for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    metrics = trainer.train_step(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    total_loss += metrics["loss"]
                    if use_wandb:
                        wandb.log({"loss": metrics["loss"], "epoch": epoch + 1})
                    else:
                        print(f"Batch loss: {metrics['loss']:.4f}")
                    
                    # Clear memory after each batch
                    del batch
                    clear_gpu_memory()
                
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
                
                # Log epoch metrics
                if use_wandb:
                    wandb.log({"avg_loss": avg_loss, "epoch": epoch + 1})
                
                # Save checkpoint
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-{epoch + 1}"))
                
                # Clear memory after each epoch
                clear_gpu_memory()
                
            except Exception as e:
                print(f"Error during epoch {epoch + 1}: {str(e)}")
                # Save checkpoint even if there's an error
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(os.path.join(output_dir, f"checkpoint-{epoch + 1}-error"))
                raise e
        
        # Save final model
        model.save_pretrained(os.path.join(output_dir, "final"))
        if use_wandb:
            run.finish()
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        if use_wandb:
            run.finish()
        raise e
    finally:
        clear_gpu_memory()

def train_dpo(
    model_name: str,
    dataset_name: str = "ultrafeedback",
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    sft_model_path: str = None,
    output_dir: str = "outputs/dpo",
    use_wandb: bool = True,
    dataloader: Optional[DataLoader] = None,
    gradient_accumulation_steps: int = 4,
    subset_size: Optional[int] = None,
    max_steps: Optional[int] = None
):
    """Train model using DPO."""
    try:
        # Initialize wandb if enabled
        if use_wandb:
            run = wandb.init(
                entity="zhao111han-stanford-university",
                project="cs224r_deeprl_final",
                name=f"dpo_{model_name.split('/')[-1]}_{dataset_name}{'_'+str(subset_size) if subset_size else ''}",
                config={
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "max_length": max_length,
                    "sft_model_path": sft_model_path,
                    "method": "dpo",
                    "effective_batch_size": batch_size * gradient_accumulation_steps,
                    "subset_size": subset_size
                }
            )
        
        # Initialize models
        model = QwenModel.from_pretrained(sft_model_path)
        ref_model = QwenModel.from_pretrained(sft_model_path)
        tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        
        # Create dataset and dataloader if not provided
        if dataloader is None:
            dataset = PreferenceDataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=max_length,
                subset_size=subset_size
            )
            dataloader = create_dataloader(dataset, batch_size=batch_size)
            
            # Print dataset information
            print(f"Training on {len(dataset)} examples from {dataset_name}" + 
                  (f" (subset: {subset_size})" if subset_size else ""))
        
        # Initialize trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            beta=0.1,  # Using 0.1 (reduced from 0.2) as in objectives.py
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_wandb=use_wandb,
            debug_mode=True  # Always enable debug mode for better diagnostics
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Train the model
        if max_steps is not None and max_steps > 0:
            # If max_steps is provided, use it instead of epochs
            trainer.train(dataloader=dataloader, max_steps=max_steps)
        else:
            # Otherwise use epochs as before
            trainer.train(dataloader=dataloader, num_epochs=num_epochs)
        
        # Save final model
        model.save_pretrained(os.path.join(output_dir, "final"))
        if use_wandb:
            run.finish()
            
    except Exception as e:
        print(f"DPO training failed: {str(e)}")
        if use_wandb and 'run' in locals():
            run.finish()
        raise e
    finally:
        clear_gpu_memory()

def train_rloo(
    model_name: str,
    dataset_name: str = "ultrafeedback",
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    sft_model_path: str = None,
    output_dir: str = "outputs/rloo",
    use_wandb: bool = True,
    dataloader: Optional[DataLoader] = None,
    gradient_accumulation_steps: int = 4,
    subset_size: Optional[int] = None,
    max_steps: Optional[int] = None,
    reward_model_epochs: int = 2,
    reward_model_lr: float = 5e-6,
    train_reward_model: bool = True,
    reward_model_path: str = None
):
    """Train model using RLOO."""
    try:
        # Initialize wandb if enabled
        if use_wandb:
            run = wandb.init(
                entity="zhao111han-stanford-university",
                project="cs224r_deeprl_final",
                name=f"rloo_{model_name.split('/')[-1]}_{dataset_name}{'_'+str(subset_size) if subset_size else ''}",
                config={
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "max_length": max_length,
                    "sft_model_path": sft_model_path,
                    "method": "rloo",
                    "effective_batch_size": batch_size * gradient_accumulation_steps,
                    "subset_size": subset_size,
                    "reward_model": "LoRA-adapted " + model_name,
                    "reward_model_epochs": reward_model_epochs if train_reward_model else 0,
                    "reward_model_lr": reward_model_lr,
                    "train_reward_model": train_reward_model,
                    "reward_model_path": reward_model_path
                }
            )
        
        # Initialize model and tokenizer from SFT path
        model = QwenModel.from_pretrained(sft_model_path)
        tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        
        # Create dataset and dataloader if not provided
        if dataloader is None:
            dataset = PreferenceDataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=max_length,
                subset_size=subset_size
            )
            dataloader = create_dataloader(dataset, batch_size=batch_size)
            
            # Print dataset information
            print(f"Training on {len(dataset)} examples from {dataset_name}" + 
                  (f" (subset: {subset_size})" if subset_size else ""))
        
        # Reward model setup
        reward_model_save_path = os.path.join(output_dir, "reward_model")
        
        # Check if we should load a pre-trained reward model
        if reward_model_path:
            print(f"Loading pre-trained reward model from {reward_model_path}")
            reward_model = RewardModel.from_pretrained(reward_model_path)
            print("Pre-trained reward model loaded successfully")
        else:
            # Initialize a new reward model
            reward_model = RewardModel(
                model_name=model_name,
                lora_r=8,
                lora_alpha=32,
                lora_dropout=0.1
            )
            print(f"Initialized LoRA-adapted reward model based on {model_name}")
            
            # Train the reward model if needed
            if train_reward_model:
                print(f"Training reward model for {reward_model_epochs} epochs...")
                # Initialize reward trainer
                reward_trainer = RewardTrainer(
                    model=reward_model,
                    learning_rate=reward_model_lr,
                    weight_decay=0.01
                )
                
                # Training reward model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                total_steps = 0
                
                for epoch in range(reward_model_epochs):
                    print(f"Reward model training epoch {epoch+1}/{reward_model_epochs}")
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
                os.makedirs(reward_model_save_path, exist_ok=True)
                reward_model.save_pretrained(reward_model_save_path)
                print(f"Reward model saved to {reward_model_save_path}")
                
                # Clear GPU memory after reward model training
                clear_gpu_memory()
            else:
                print("Skipping reward model training, using untrained model (not recommended)")
        
        # Set reward model to evaluation mode for RLOO training
        reward_model.eval()
        
        print("Starting RLOO training...")
        # Initialize RLOO trainer
        trainer = RLOOTrainer(
            model=model,
            reward_model=reward_model,
            learning_rate=learning_rate,
            num_samples=4,  # Number of samples for RLOO
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_wandb=use_wandb,
            debug_mode=True  # Always enable debug mode for better diagnostics
        )
        
        # Train the model
        if max_steps is not None and max_steps > 0:
            # If max_steps is provided, use it instead of epochs
            trainer.train(dataloader=dataloader, max_steps=max_steps)
        else:
            # Otherwise use epochs as before
            trainer.train(dataloader=dataloader, num_epochs=num_epochs)
        
        # Save final model
        model.save_pretrained(os.path.join(output_dir, "final"))
        if use_wandb:
            run.finish()
            
    except Exception as e:
        print(f"RLOO training failed: {str(e)}")
        if use_wandb and 'run' in locals():
            run.finish()
        raise e
    finally:
        clear_gpu_memory()

def evaluate_model(
    model_path: str,
    num_prompts: int,
    output_dir: str,
    use_wandb: bool = True
):
    """Evaluate model performance."""
    # Initialize wandb if enabled
    if use_wandb:
        run = wandb.init(
            # Set the entity (usually your username or team name)
            entity="zhao111han-stanford-university",
            # Set the project
            project="cs224r_deeprl_final",
            # Name this run for better tracking
            name=f"eval_{model_path.split('/')[-2]}",
            # Track hyperparameters and run metadata
            config={
                "model_path": model_path,
                "num_prompts": num_prompts,
                "method": "eval"
            }
        )
    
    # Initialize model
    model = QwenModel.from_pretrained(model_path)
    
    # Run evaluation
    metrics = run_evaluation(model, num_prompts)
    
    # Print metrics
    print("\nEvaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    if use_wandb:
        # Log all metrics at once
        wandb.log(metrics)
        # Finish the run
        run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["sft", "dpo", "rloo", "eval"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str)  # No default, will be set based on method
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--sft_model_path", type=str)
    parser.add_argument("--model_path", type=str, help="Path to model for evaluation")
    parser.add_argument("--num_prompts", type=int, default=100, help="Number of prompts for evaluation")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--subset_size", type=int, default=None, help="Number of examples to use from dataset (for debugging/testing)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of training steps (overrides num_epochs if set)")
    # Add reward model training arguments
    parser.add_argument("--reward_model_epochs", type=int, default=2, help="Number of epochs for reward model training")
    parser.add_argument("--reward_model_lr", type=float, default=5e-6, help="Learning rate for reward model training")
    parser.add_argument("--no_train_reward_model", action="store_true", help="Skip reward model training (use existing or untrained model)")
    parser.add_argument("--reward_model_path", type=str, help="Path to a pre-trained reward model to use instead of training one")
    args = parser.parse_args()
    
    # Set default dataset based on method if not specified
    if args.dataset_name is None:
        if args.method == "sft":
            args.dataset_name = "smoltalk"
        else:  # dpo or rloo
            args.dataset_name = "ultrafeedback"
    
    if args.method == "sft":
        train_sft(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            output_dir=args.output_dir or "outputs/sft",
            use_wandb=args.use_wandb,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
    elif args.method == "dpo":
        if not args.sft_model_path:
            raise ValueError("sft_model_path is required for DPO training")
        train_dpo(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            sft_model_path=args.sft_model_path,
            output_dir=args.output_dir or "outputs/dpo",
            use_wandb=args.use_wandb,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            subset_size=args.subset_size,
            max_steps=args.max_steps
        )
    elif args.method == "rloo":
        if not args.sft_model_path:
            raise ValueError("sft_model_path is required for RLOO training")
        train_rloo(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            sft_model_path=args.sft_model_path,
            output_dir=args.output_dir or "outputs/rloo",
            use_wandb=args.use_wandb,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            subset_size=args.subset_size,
            max_steps=args.max_steps,
            reward_model_epochs=args.reward_model_epochs,
            reward_model_lr=args.reward_model_lr,
            train_reward_model=not args.no_train_reward_model,
            reward_model_path=args.reward_model_path
        )
    elif args.method == "eval":
        if not args.model_path:
            raise ValueError("model_path is required for evaluation")
        evaluate_model(
            model_path=args.model_path,
            num_prompts=args.num_prompts,
            output_dir=args.output_dir or f"outputs/eval/{args.model_path.split('/')[-2]}",
            use_wandb=args.use_wandb
        ) 