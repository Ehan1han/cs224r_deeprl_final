from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

class RewardModel(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ):
        """
        Initialize the reward model with LoRA adaptation.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on
            lora_r: Rank of LoRA adaptation matrices
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
        """
        super().__init__()
        self.device = device
        
        # Initialize tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure there's a padding token defined
        if self.tokenizer.pad_token is None:
            print(f"No padding token found. Setting pad_token to eos_token ({self.tokenizer.eos_token})")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model and add a classification head
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
            pad_token_id=self.tokenizer.pad_token_id  # Pass the pad token ID
        )
        
        # Ensure the model's config has the padding token ID set
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Define LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Target attention layers
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(base_model, peft_config)
        self.model.to(device)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the reward model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Reward values
        """
        # Ensure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits.squeeze(-1)
    
    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Bradley-Terry loss.
        
        Args:
            chosen_ids: Chosen response token IDs
            rejected_ids: Rejected response token IDs
            chosen_mask: Chosen response attention mask
            rejected_mask: Rejected response attention mask
            
        Returns:
            Loss tensor and metrics dictionary
        """
        # Get rewards for chosen and rejected responses
        chosen_rewards = self.forward(chosen_ids, chosen_mask)
        rejected_rewards = self.forward(rejected_ids, rejected_mask)
        
        # Compute Bradley-Terry loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # Compute accuracy
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item()
        }
        
        return loss, metrics
    
    def save_pretrained(self, path: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Load a saved model."""
        # Initialize with default values
        model = cls(device=device)
        
        # Load the PEFT model
        if PeftModel.is_peft_model(path):
            model.model = PeftModel.from_pretrained(path)
        else:
            # For backward compatibility, load as regular model if not a PEFT model
            model.model = AutoModelForSequenceClassification.from_pretrained(path)
        
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Ensure padding token is set
        if model.tokenizer.pad_token is None:
            print(f"No padding token found during loading. Setting pad_token to eos_token.")
            model.tokenizer.pad_token = model.tokenizer.eos_token
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
            
        # Update model config
        model.model.config.pad_token_id = model.tokenizer.pad_token_id
        
        model.model.to(device)
        return model

class RewardTrainer:
    def __init__(
        self,
        model: RewardModel,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01
    ):
        """
        Initialize reward model trainer.
        
        Args:
            model: Reward model instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            chosen_ids: Chosen response token IDs
            rejected_ids: Rejected response token IDs
            chosen_mask: Chosen response attention mask
            rejected_mask: Rejected response attention mask
            
        Returns:
            Dictionary containing loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, metrics = self.model.compute_loss(
            chosen_ids=chosen_ids,
            rejected_ids=rejected_ids,
            chosen_mask=chosen_mask,
            rejected_mask=rejected_mask
        )
        
        loss.backward()
        self.optimizer.step()
        
        return metrics
