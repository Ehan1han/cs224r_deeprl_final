from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

class QwenModel(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Qwen model wrapper.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on
        """
        super().__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure generation parameters
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.max_new_tokens = 128  # Set default max_new_tokens
        self.model.generation_config.max_length = None  # Disable max_length to avoid conflicts
        
        self.model.to(device)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for computing loss
            
        Returns:
            Model outputs including loss if labels are provided
        """
        # Move inputs to the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text from the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        # Move inputs to the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def save_pretrained(self, path: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Load a saved model."""
        model = cls(device=device)
        model.model = AutoModelForCausalLM.from_pretrained(path)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        model.model.to(device)
        return model

def compute_loss(
    outputs: CausalLMOutputWithPast,
    labels: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute loss while masking out query tokens.
    
    Args:
        outputs: Model outputs
        labels: Ground truth labels
        attention_mask: Attention mask
        
    Returns:
        Loss tensor
    """
    # Shift labels and logits for next-token prediction
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    
    # Compute loss only on response tokens
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Apply attention mask
    loss = loss.view(shift_labels.size())
    loss = (loss * shift_attention_mask).sum() / shift_attention_mask.sum()
    
    return loss