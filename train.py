import os
import argparse
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
import json

# Set multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

from rl_llm.data import PreferenceDataset, create_dataloader
from rl_llm.model_utils import QwenModel
from rl_llm.objectives import SFTTrainer, DPOTrainer, RLOOTrainer
from rl_llm.reward import RewardModel, RewardTrainer
from rl_llm.evaluation import run_evaluation

def train_sft(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    dataset_name: str = "smoltalk",
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    output_dir: str = "outputs/sft",
    use_wandb: bool = True
):
    """Train model using SFT."""
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="rl_llm_final",
            entity="yhanzhao",
            config={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "max_length": max_length,
                "method": "sft"
            }
        )
    
    # Initialize model and tokenizer
    model = QwenModel(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataset and dataloader
    dataset = PreferenceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length
    )
    dataloader = create_dataloader(dataset, batch_size=batch_size)
    
    # Initialize trainer
    trainer = SFTTrainer(model, learning_rate=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
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
                wandb.log(metrics)
            else:
                print(f"Batch loss: {metrics['loss']:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(output_dir, f"checkpoint-{epoch + 1}"))
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    if use_wandb:
        wandb.finish()

def train_dpo(
    model_name: str,
    dataset_name: str = "ultrafeedback",
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    sft_model_path: str = None,
    output_dir: str = "outputs/dpo",
    use_wandb: bool = True
):
    """Train model using DPO."""
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="rl_llm_final",
            entity="yhanzhao",
            config={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "max_length": max_length,
                "method": "dpo"
            }
        )
    
    # Initialize models
    model = QwenModel.from_pretrained(sft_model_path)
    ref_model = QwenModel.from_pretrained(sft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    
    # Create dataset and dataloader
    dataset = PreferenceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length
    )
    dataloader = create_dataloader(dataset, batch_size=batch_size)
    
    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=0.1,
        learning_rate=learning_rate
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = trainer.train_step(
                prompt_ids=batch["chosen_input_ids"],
                chosen_ids=batch["chosen_input_ids"],
                rejected_ids=batch["rejected_input_ids"],
                prompt_mask=batch["chosen_attention_mask"],
                chosen_mask=batch["chosen_attention_mask"],
                rejected_mask=batch["rejected_attention_mask"]
            )
            total_loss += metrics["loss"]
            if use_wandb:
                wandb.log(metrics)
            else:
                print(f"Batch loss: {metrics['loss']:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(output_dir, f"checkpoint-{epoch + 1}"))
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    if use_wandb:
        wandb.finish()

def train_rloo(
    model_name: str,
    dataset_name: str = "ultrafeedback",
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    output_dir: str = "outputs/rloo",
    use_wandb: bool = True
):
    """Train model using RLOO."""
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="rl_llm_final",
            entity="yhanzhao",
            config={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "max_length": max_length,
                "method": "rloo"
            }
        )
    
    # Initialize model and tokenizer
    model = QwenModel(model_name)
    tokenizer = model.tokenizer
    
    # Create dataset and dataloader
    dataset = PreferenceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length,
        subset_size=100
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize reward model
    reward_model = RewardModel(model_name)
    
    # Initialize trainer
    trainer = RLOOTrainer(
        model=model,
        reward_model=reward_model,
        learning_rate=learning_rate,
        num_samples=4
    )
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Use correct keys for UltraFeedback
            if "chosen_input_ids" in batch:
                input_ids = batch["chosen_input_ids"]
                attention_mask = batch["chosen_attention_mask"]
            else:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
            metrics = trainer.train_step(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            total_loss += metrics["loss"]
            if use_wandb:
                wandb.log({"rloo_loss": metrics["loss"]})
            else:
                print(f"Batch loss: {metrics['loss']:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(output_dir, f"checkpoint-{epoch + 1}"))
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    if use_wandb:
        wandb.finish()

def evaluate_model(
    model_path: str,
    num_prompts: int,
    output_dir: str,
    use_wandb: bool = True
):
    """Evaluate model performance."""
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="rl_llm_final",
            entity="yhanzhao",
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
        wandb.log(metrics)
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["sft", "dpo", "rloo"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str)  # No default, will be set based on method
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--sft_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--subset_size", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
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
            use_wandb=args.use_wandb
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
            use_wandb=args.use_wandb
        )
    elif args.method == "rloo":
        train_rloo(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            output_dir=args.output_dir or "outputs/rloo",
            use_wandb=args.use_wandb
        ) 