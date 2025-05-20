import os
import torch
from transformers import AutoTokenizer
from rl_llm.data import PreferenceDataset, create_dataloader
from rl_llm.model_utils import QwenModel
from rl_llm.reward import RewardModel
from rl_llm.objectives import RLOOTrainer
from tqdm import tqdm

def test_rloo():
    print("Starting RLOO test...")
    
    # Test parameters
    test_size = 10
    print(f"Using test size: {test_size} samples")
    
    # Initialize model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    model = QwenModel(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create small test dataset
    dataset = PreferenceDataset(
        dataset_name="ultrafeedback",
        tokenizer=tokenizer,
        subset_size=test_size
    )
    dataloader = create_dataloader(dataset, batch_size=2)
    
    # Initialize reward model
    reward_model = RewardModel(model_name)
    
    # Initialize RLOO trainer
    trainer = RLOOTrainer(
        model=model,
        reward_model=reward_model,
        learning_rate=1e-5,
        num_samples=4
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Train step
            metrics = trainer.train_step(
                prompt_ids=batch["chosen_input_ids"],
                chosen_ids=batch["chosen_input_ids"],
                rejected_ids=batch["rejected_input_ids"],
                prompt_mask=batch["chosen_attention_mask"],
                chosen_mask=batch["chosen_attention_mask"],
                rejected_mask=batch["rejected_attention_mask"]
            )
            
            total_loss += metrics["loss"]
            print(f"Batch loss: {metrics['loss']:.4f}, Avg reward: {metrics['avg_reward']:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        output_dir = "outputs/test/rloo"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(output_dir, f"checkpoint-{epoch + 1}"))
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    print("RLOO test completed!")

if __name__ == "__main__":
    test_rloo() 