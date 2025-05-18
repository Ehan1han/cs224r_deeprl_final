from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the reward model.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on
        """
        super().__init__()
        self.device = device
        
        # Load base model and add a classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        model = cls(device=device)
        model.model = AutoModelForSequenceClassification.from_pretrained(path)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
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
