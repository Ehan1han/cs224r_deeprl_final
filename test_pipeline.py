import os
import torch
import wandb
import gc
from transformers import AutoTokenizer
from rl_llm.data import PreferenceDataset, create_dataloader
from rl_llm.model_utils import QwenModel
from rl_llm.objectives import SFTTrainer, DPOTrainer, RLOOTrainer
from train import train_sft, train_dpo, train_rloo

def test_pipeline(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    test_size: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    num_epochs: int = 2,
    max_length: int = 512,
    output_dir: str = "outputs/test",
    use_wandb: bool = False,
    gradient_accumulation_steps: int = 4
):
    """
    Run a test version of the complete training pipeline with a small dataset.
    
    Args:
        model_name: Base model to use
        test_size: Number of samples to use for testing
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        num_epochs: Number of epochs to train
        max_length: Maximum sequence length
        output_dir: Base directory for outputs
        use_wandb: Whether to use Weights & Biases logging
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    print("Starting test pipeline...")
    print(f"Using test size: {test_size} samples")
    print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # Create output directories
    sft_dir = os.path.join(output_dir, "sft")
    dpo_dir = os.path.join(output_dir, "dpo")
    rloo_dir = os.path.join(output_dir, "rloo")
    
    try:
        # 1. Test SFT Training
        print("\nTesting SFT training...")
        # Initialize tokenizer for dataset creation
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create small test dataset
        sft_dataset = PreferenceDataset(
            dataset_name="smoltalk",
            tokenizer=tokenizer,
            max_length=max_length,
            subset_size=test_size
        )
        sft_dataloader = create_dataloader(sft_dataset, batch_size=batch_size)
        
        train_sft(
            model_name=model_name,
            dataset_name="smoltalk",
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            max_length=max_length,
            output_dir=sft_dir,
            use_wandb=use_wandb,
            dataloader=sft_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Explicitly clear GPU memory after SFT
        print("\nClearing GPU memory after SFT training...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 2. Test DPO Training
        print("\nTesting DPO training...")
        # Create small test dataset for DPO
        dpo_dataset = PreferenceDataset(
            dataset_name="ultrafeedback",
            tokenizer=tokenizer,
            max_length=max_length,
            subset_size=test_size
        )
        dpo_dataloader = create_dataloader(dpo_dataset, batch_size=batch_size)
        
        train_dpo(
            model_name=model_name,
            dataset_name="ultrafeedback",
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            max_length=max_length,
            sft_model_path=os.path.join(sft_dir, "final"),
            output_dir=dpo_dir,
            use_wandb=use_wandb,
            dataloader=dpo_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Explicitly clear GPU memory after DPO
        print("\nClearing GPU memory after DPO training...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 3. Test RLOO Training
        print("\nTesting RLOO training...")
        # Create small test dataset for RLOO
        rloo_dataset = PreferenceDataset(
            dataset_name="ultrafeedback",
            tokenizer=tokenizer,
            max_length=max_length,
            subset_size=test_size
        )
        rloo_dataloader = create_dataloader(rloo_dataset, batch_size=batch_size)
        
        train_rloo(
            model_name=model_name,
            dataset_name="ultrafeedback",
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            max_length=max_length,
            output_dir=rloo_dir,
            use_wandb=use_wandb,
            dataloader=rloo_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        print("\nTest pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError in test pipeline: {str(e)}")
        raise e
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    # Run the test pipeline
    test_pipeline() 