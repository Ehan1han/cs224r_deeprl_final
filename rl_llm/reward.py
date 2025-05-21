from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-base",  # Using RoBERTa as a more appropriate backbone
        hidden_size: int = 768,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the reward model.
        
        Args:
            model_name: Name of the backbone model to load
            hidden_size: Size of hidden layers (should match the backbone model)
            dropout: Dropout rate
            device: Device to load the model on
        """
        super().__init__()
        self.device = device
        
        # Load model and tokenizer with trust_remote_code=True for Qwen models
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Update hidden size if needed
        if hasattr(self.backbone.config, 'hidden_size'):
            hidden_size = self.backbone.config.hidden_size
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        self.to(device)
        
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
        # Get sequence embeddings from backbone model
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding for reward prediction
        cls_embedding = outputs.last_hidden_state[:, 0]
        
        # Compute reward
        reward = self.reward_head(cls_embedding).squeeze(-1)
        return reward
    
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
        self.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Load a saved model."""
        model = cls(device=device)
        model.backbone = AutoModel.from_pretrained(path, trust_remote_code=True)
        model.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model.to(device)
        return model

class RewardTrainer:
    def __init__(
        self,
        model: RewardModel,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize reward model trainer.
        
        Args:
            model: Reward model instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for learning rate schedule
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
    def get_current_lr(self) -> float:
        """Get the current learning rate with warmup schedule."""
        if self.current_step < self.warmup_steps:
            return self.learning_rate * (self.current_step / self.warmup_steps)
        return self.learning_rate
        
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
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_current_lr()
        
        # Compute loss and metrics
        loss, metrics = self.model.compute_loss(
            chosen_ids=chosen_ids,
            rejected_ids=rejected_ids,
            chosen_mask=chosen_mask,
            rejected_mask=rejected_mask
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update step counter
        self.current_step += 1
        
        return metrics
