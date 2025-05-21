from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import QwenModel, compute_loss
import wandb

class SFTTrainer:
    def __init__(
        self,
        model: QwenModel,
        learning_rate: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4
    ):
        """
        Initialize SFT trainer.
        
        Args:
            model: Qwen model instance
            learning_rate: Learning rate for optimizer
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_step = 0
        
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_response_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            prompt_response_mask: Mask distinguishing prompt (0) from response (1) tokens
            labels: Labels for computing loss
            
        Returns:
            Dictionary containing loss value
        """
        self.model.train()
        
        # Only zero the gradients for the first step in the accumulation cycle
        if self.current_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Pass both masks separately to compute_loss
        loss = compute_loss(
            outputs=outputs, 
            labels=labels, 
            attention_mask=attention_mask,
            prompt_response_mask=prompt_response_mask
        )
        
        # Scale the loss by the number of accumulation steps
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        
        # Only perform optimization step after accumulating gradients
        if (self.current_step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
        
        self.current_step += 1
        
        return {"loss": loss.item()}

class DPOTrainer:
    def __init__(
        self,
        model: QwenModel,
        ref_model: QwenModel,
        beta: float = 0.1,  # Reduced from 0.2 to 0.1 to make updates more gentle
        learning_rate: float = 1e-5,
        max_grad_norm: float = 0.5,  # Reduced from 1.0 to 0.5 for more stability
        gradient_accumulation_steps: int = 4,
        use_wandb: bool = True,
        debug_mode: bool = True  # Added debug mode for additional diagnostics
    ):
        """
        Initialize DPO trainer.
        
        Args:
            model: Policy model to train
            ref_model: Reference model (typically SFT model)
            beta: Temperature parameter for DPO (lower values = more conservative updates)
            learning_rate: Learning rate for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            gradient_accumulation_steps: Number of steps to accumulate gradients
            use_wandb: Whether to use wandb for logging
            debug_mode: Whether to log additional debugging information
        """
        self.model = model
        self.ref_model = ref_model
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.device = next(model.parameters()).device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_step = 0
        self.use_wandb = use_wandb
        self.debug_mode = debug_mode
        
    def train_step(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        chosen_prompt_response_mask: Optional[torch.Tensor] = None,
        rejected_prompt_response_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform a single DPO training step.
        
        Args:
            prompt_ids: Input prompt token IDs
            chosen_ids: Chosen response token IDs
            rejected_ids: Rejected response token IDs
            prompt_mask: Prompt attention mask
            chosen_mask: Chosen response attention mask
            rejected_mask: Rejected response attention mask
            chosen_prompt_response_mask: Optional mask for chosen (0=prompt, 1=response)
            rejected_prompt_response_mask: Optional mask for rejected (0=prompt, 1=response)
            
        Returns:
            Dictionary containing loss value and metrics
        """
        self.model.train()
        self.ref_model.eval()
        
        # Only zero the gradients for the first step in the accumulation cycle
        if self.current_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
        
        # Move inputs to device
        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        chosen_mask = chosen_mask.to(self.device)
        rejected_mask = rejected_mask.to(self.device)
        
        # Get policy logits
        chosen_outputs = self.model(
            input_ids=chosen_ids,
            attention_mask=chosen_mask,
            labels=chosen_ids
        )
        rejected_outputs = self.model(
            input_ids=rejected_ids,
            attention_mask=rejected_mask,
            labels=rejected_ids
        )
        
        # Get reference logits
        with torch.no_grad():
            chosen_ref_outputs = self.ref_model(
                input_ids=chosen_ids,
                attention_mask=chosen_mask,
                labels=chosen_ids
            )
            rejected_ref_outputs = self.ref_model(
                input_ids=rejected_ids,
                attention_mask=rejected_mask,
                labels=rejected_ids
            )
        
        # Compute policy and reference logits
        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits
        chosen_ref_logits = chosen_ref_outputs.logits
        rejected_ref_logits = rejected_ref_outputs.logits
        
        # Shift logits and labels for next-token prediction
        chosen_logits = chosen_logits[..., :-1, :].contiguous()
        rejected_logits = rejected_logits[..., :-1, :].contiguous()
        chosen_ref_logits = chosen_ref_logits[..., :-1, :].contiguous()
        rejected_ref_logits = rejected_ref_logits[..., :-1, :].contiguous()
        
        chosen_labels = chosen_ids[..., 1:].contiguous()
        rejected_labels = rejected_ids[..., 1:].contiguous()
        
        # Compute log probabilities
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
        chosen_ref_log_probs = F.log_softmax(chosen_ref_logits, dim=-1)
        rejected_ref_log_probs = F.log_softmax(rejected_ref_logits, dim=-1)
        
        # Gather log probabilities for the target tokens
        chosen_log_probs = chosen_log_probs.gather(-1, chosen_labels.unsqueeze(-1)).squeeze(-1)
        rejected_log_probs = rejected_log_probs.gather(-1, rejected_labels.unsqueeze(-1)).squeeze(-1)
        chosen_ref_log_probs = chosen_ref_log_probs.gather(-1, chosen_labels.unsqueeze(-1)).squeeze(-1)
        rejected_ref_log_probs = rejected_ref_log_probs.gather(-1, rejected_labels.unsqueeze(-1)).squeeze(-1)
        
        # Apply masks
        chosen_mask = chosen_mask[..., 1:].contiguous()
        rejected_mask = rejected_mask[..., 1:].contiguous()
        
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-8
        chosen_mask_sum = chosen_mask.sum(dim=-1).clamp(min=epsilon)
        rejected_mask_sum = rejected_mask.sum(dim=-1).clamp(min=epsilon)
        
        # Compute mean log probabilities over sequence length
        chosen_log_probs = (chosen_log_probs * chosen_mask).sum(dim=-1) / chosen_mask_sum
        rejected_log_probs = (rejected_log_probs * rejected_mask).sum(dim=-1) / rejected_mask_sum
        chosen_ref_log_probs = (chosen_ref_log_probs * chosen_mask).sum(dim=-1) / chosen_mask_sum
        rejected_ref_log_probs = (rejected_ref_log_probs * rejected_mask).sum(dim=-1) / rejected_mask_sum
        
        # Compute log probability ratios (with additional numerical stability checks)
        chosen_ratio = chosen_log_probs - chosen_ref_log_probs
        rejected_ratio = rejected_log_probs - rejected_ref_log_probs
        
        # Clamp ratios to avoid numerical instability
        max_ratio_value = 50.0  # Prevent exp(x) overflow
        chosen_ratio = torch.clamp(chosen_ratio, min=-max_ratio_value, max=max_ratio_value)
        rejected_ratio = torch.clamp(rejected_ratio, min=-max_ratio_value, max=max_ratio_value)
        
        # Compute DPO loss: -E[log σ(β * (log πθ(yw|x)/πref(yw|x) - log πθ(yl|x)/πref(yl|x)))]
        # Using logsigmoid for numerical stability
        loss = -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio)).mean()
        
        # Handle NaN loss (use a fallback value if loss is NaN)
        if torch.isnan(loss).item():
            print("NaN loss detected, using fallback value")
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        # Scale the loss by the number of accumulation steps
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        
        # Only perform optimization step after accumulating gradients
        if (self.current_step + 1) % self.gradient_accumulation_steps == 0:
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Log gradient norm if using wandb
            if self.use_wandb:
                wandb.log({"grad_norm": grad_norm.item()}, step=self.current_step)
        
        self.current_step += 1
        
        # Ensure no NaN values in metrics
        metrics = {
            "loss": loss.item() if not torch.isnan(loss).item() else 1.0,
            "chosen_ratio": chosen_ratio.mean().item() if not torch.isnan(chosen_ratio).any().item() else 0.0,
            "rejected_ratio": rejected_ratio.mean().item() if not torch.isnan(rejected_ratio).any().item() else 0.0,
            "chosen_log_probs": chosen_log_probs.mean().item() if not torch.isnan(chosen_log_probs).any().item() else 0.0,
            "rejected_log_probs": rejected_log_probs.mean().item() if not torch.isnan(rejected_log_probs).any().item() else 0.0,
            "chosen_ref_log_probs": chosen_ref_log_probs.mean().item() if not torch.isnan(chosen_ref_log_probs).any().item() else 0.0,
            "rejected_ref_log_probs": rejected_ref_log_probs.mean().item() if not torch.isnan(rejected_ref_log_probs).any().item() else 0.0
        }
        
        # Log metrics to wandb if enabled
        if self.use_wandb:
            wandb.log(metrics, step=self.current_step)
            
        return metrics

    def train(self, dataloader=None, num_epochs=1, max_steps=None):
        """
        Train the model on the provided dataloader.
        
        Args:
            dataloader: DataLoader containing training data
            num_epochs: Number of epochs to train for
            max_steps: Maximum number of steps to train for (overrides num_epochs if provided)
        
        Returns:
            Dictionary containing training metrics
        """
        if dataloader is None:
            raise ValueError("dataloader must be provided")
        
        self.model.train()
        self.ref_model.eval()
        
        device = next(self.model.parameters()).device
        step_count = 0
        total_loss = 0
        total_chosen_ratio = 0
        total_rejected_ratio = 0
        total_steps = 0
        
        # Training loop
        from tqdm import tqdm
        
        if max_steps is not None:
            # If max_steps is provided, loop until reaching that many steps
            progress_bar = tqdm(total=max_steps, desc="Training")
            # Calculate true epoch based on steps and dataset size
            steps_per_epoch = len(dataloader)
            while step_count < max_steps:
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Get batch size
                    batch_size = batch["chosen_input_ids"].size(0)
                    
                    # Use the prompt_response_mask from the dataset instead of computing it
                    prompt_input_ids = batch["chosen_input_ids"].clone()
                    prompt_attention_mask = batch["chosen_attention_mask"].clone()
                    
                    # Use the masks from the dataset that properly distinguish prompt and response tokens
                    chosen_attention_mask = batch["chosen_attention_mask"].clone() * batch["chosen_prompt_response_mask"].clone()
                    rejected_attention_mask = batch["rejected_attention_mask"].clone() * batch["rejected_prompt_response_mask"].clone()
                    
                    if self.debug_mode and step_count % 50 == 0:
                        print("\n--- Prompt/Response Analysis ---")
                        print(f"Batch size: {batch_size}")
                        print(f"Using dataset-provided prompt/response masks")
                        
                        # Check mask shapes
                        print(f"Chosen mask shape: {chosen_attention_mask.shape}")
                        print(f"Rejected mask shape: {rejected_attention_mask.shape}")
                        
                        # Check correct masking for each example individually
                        mask_check_results = []
                        for b in range(min(batch["chosen_input_ids"].size(0), 3)):  # Check up to 3 examples
                            # Count non-zero elements in prompt and response regions
                            prompt_tokens = (batch["chosen_prompt_response_mask"][b] == 0).sum().item()
                            response_tokens = (batch["chosen_prompt_response_mask"][b] == 1).sum().item()
                            mask_check_results.append(f"Example {b}: prompt tokens={prompt_tokens}, response tokens={response_tokens}")
                        
                        print("Mask check (prompt=0, response=1):")
                        for result in mask_check_results:
                            print(f"  {result}")
                        print("-------------------------------")
                    
                    # Forward and backward pass
                    metrics = self.train_step(
                        prompt_ids=prompt_input_ids,
                        chosen_ids=batch["chosen_input_ids"],
                        rejected_ids=batch["rejected_input_ids"],
                        prompt_mask=prompt_attention_mask,
                        chosen_mask=chosen_attention_mask,
                        rejected_mask=rejected_attention_mask,
                        chosen_prompt_response_mask=batch["chosen_prompt_response_mask"],
                        rejected_prompt_response_mask=batch["rejected_prompt_response_mask"]
                    )
                    
                    total_loss += metrics["loss"]
                    total_chosen_ratio += metrics.get("chosen_ratio", 0)
                    total_rejected_ratio += metrics.get("rejected_ratio", 0)
                    total_steps += 1
                    step_count += 1
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": metrics["loss"],
                        "chosen_ratio": metrics.get("chosen_ratio", 0),
                        "rejected_ratio": metrics.get("rejected_ratio", 0)
                    })
                    
                    # Log epoch-level metrics if using wandb
                    if self.use_wandb:
                        # Calculate current epoch as a float to show partial progress
                        current_epoch = step_count / steps_per_epoch
                        wandb.log({
                            "epoch": current_epoch,
                            "epoch_avg_loss": total_loss / total_steps,
                            "epoch_avg_chosen_ratio": total_chosen_ratio / total_steps,
                            "epoch_avg_rejected_ratio": total_rejected_ratio / total_steps,
                            "learning_rate": self.optimizer.param_groups[0]["lr"]
                        }, step=self.current_step)
                    
                    if step_count >= max_steps:
                        break
                
        else:
            # Train for a fixed number of epochs
            for epoch in range(num_epochs):
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
                epoch_loss = 0
                epoch_steps = 0
                
                for batch in progress_bar:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Get batch size
                    batch_size = batch["chosen_input_ids"].size(0)
                    
                    # Use the prompt_response_mask from the dataset instead of computing it
                    prompt_input_ids = batch["chosen_input_ids"].clone()
                    prompt_attention_mask = batch["chosen_attention_mask"].clone()
                    
                    # Use the masks from the dataset that properly distinguish prompt and response tokens
                    chosen_attention_mask = batch["chosen_attention_mask"].clone() * batch["chosen_prompt_response_mask"].clone()
                    rejected_attention_mask = batch["rejected_attention_mask"].clone() * batch["rejected_prompt_response_mask"].clone()
                    
                    if self.debug_mode and step_count % 50 == 0:
                        print("\n--- Prompt/Response Analysis ---")
                        print(f"Batch size: {batch_size}")
                        print(f"Using dataset-provided prompt/response masks")
                        
                        # Check mask shapes
                        print(f"Chosen mask shape: {chosen_attention_mask.shape}")
                        print(f"Rejected mask shape: {rejected_attention_mask.shape}")
                        
                        # Check correct masking for each example individually
                        mask_check_results = []
                        for b in range(min(batch["chosen_input_ids"].size(0), 3)):  # Check up to 3 examples
                            # Count non-zero elements in prompt and response regions
                            prompt_tokens = (batch["chosen_prompt_response_mask"][b] == 0).sum().item()
                            response_tokens = (batch["chosen_prompt_response_mask"][b] == 1).sum().item()
                            mask_check_results.append(f"Example {b}: prompt tokens={prompt_tokens}, response tokens={response_tokens}")
                        
                        print("Mask check (prompt=0, response=1):")
                        for result in mask_check_results:
                            print(f"  {result}")
                        print("-------------------------------")
                    
                    # Forward and backward pass
                    metrics = self.train_step(
                        prompt_ids=prompt_input_ids,
                        chosen_ids=batch["chosen_input_ids"],
                        rejected_ids=batch["rejected_input_ids"],
                        prompt_mask=prompt_attention_mask,
                        chosen_mask=chosen_attention_mask,
                        rejected_mask=rejected_attention_mask,
                        chosen_prompt_response_mask=batch["chosen_prompt_response_mask"],
                        rejected_prompt_response_mask=batch["rejected_prompt_response_mask"]
                    )
                    
                    total_loss += metrics["loss"]
                    total_chosen_ratio += metrics.get("chosen_ratio", 0)
                    total_rejected_ratio += metrics.get("rejected_ratio", 0)
                    total_steps += 1
                    step_count += 1
                    epoch_loss += metrics["loss"]
                    epoch_steps += 1
                    
                    progress_bar.set_postfix({
                        "loss": metrics["loss"],
                        "chosen_ratio": metrics.get("chosen_ratio", 0),
                        "rejected_ratio": metrics.get("rejected_ratio", 0)
                    })
                
                # Log epoch-level metrics if using wandb
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "epoch_avg_loss": epoch_loss / epoch_steps,
                        "epoch_avg_chosen_ratio": total_chosen_ratio / total_steps,
                        "epoch_avg_rejected_ratio": total_rejected_ratio / total_steps,
                        "learning_rate": self.optimizer.param_groups[0]["lr"]
                    }, step=self.current_step)
        
        # Return average metrics
        return {
            "loss": total_loss / total_steps,
            "chosen_ratio": total_chosen_ratio / total_steps,
            "rejected_ratio": total_rejected_ratio / total_steps
        }

class RLOOTrainer:
    def __init__(
        self,
        model: QwenModel,
        reward_model: nn.Module,
        learning_rate: float = 1e-5,
        num_samples: int = 4,
        max_grad_norm: float = 0.5,  # Reduced from 1.0 to 0.5 for more stability
        gradient_accumulation_steps: int = 4,
        use_wandb: bool = True,
        debug_mode: bool = True  # Added debug mode for additional diagnostics
    ):
        """
        Initialize RLOO trainer.
        
        Args:
            model: Policy model to train
            reward_model: Reward model for computing rewards
            learning_rate: Learning rate for optimizer
            num_samples: Number of samples for RLOO (k in the formula)
            max_grad_norm: Maximum gradient norm for clipping
            gradient_accumulation_steps: Number of steps to accumulate gradients
            use_wandb: Whether to use wandb for logging
            debug_mode: Whether to log additional debugging information
        """
        self.model = model
        self.reward_model = reward_model
        
        # Freeze reward model parameters
        for param in self.reward_model.parameters():
            param.requires_grad = False
            
        self.num_samples = num_samples
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.device = next(model.parameters()).device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_step = 0
        self.use_wandb = use_wandb
        self.debug_mode = debug_mode
        
    def train_step(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        chosen_prompt_response_mask: Optional[torch.Tensor] = None,
        rejected_prompt_response_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform a single RLOO training step.
        
        Args:
            prompt_ids: Input prompt token IDs
            chosen_ids: Chosen response token IDs
            rejected_ids: Rejected response token IDs
            prompt_mask: Prompt attention mask
            chosen_mask: Chosen response attention mask
            rejected_mask: Rejected response attention mask
            chosen_prompt_response_mask: Optional mask for chosen (0=prompt, 1=response)
            rejected_prompt_response_mask: Optional mask for rejected (0=prompt, 1=response)
            
        Returns:
            Dictionary containing loss value
        """
        self.model.train()
        self.reward_model.eval()
        
        # Only zero the gradients for the first step in the accumulation cycle
        if self.current_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
        
        # Move inputs to device
        prompt_ids = prompt_ids.to(self.device)
        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)
        chosen_mask = chosen_mask.to(self.device)
        rejected_mask = rejected_mask.to(self.device)
        
        batch_size = prompt_ids.size(0)
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_rewards = []
        
        # Process each example in the batch
        for i in range(batch_size):
            # Generate k samples for the current input
            samples = []
            sample_rewards = []
            sample_masks = []
            
            # Get prompt length from the non-masked part of the prompt
            prompt_length = prompt_ids.size(1)
            
            # Generate k samples
            for _ in range(self.num_samples):
                try:
                    with torch.no_grad():
                        sample_ids = self.model.generate(
                            input_ids=prompt_ids[i:i+1],
                            attention_mask=prompt_mask[i:i+1],
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )
                        sample_reward = self.reward_model(sample_ids)
                        samples.append(sample_ids)
                        sample_rewards.append(sample_reward)
                        
                        # Create attention mask for the sample with 0 for prompt tokens, 1 for response
                        sample_mask = torch.ones_like(sample_ids)
                        sample_mask[:, :prompt_length] = 0  # Zero out prompt tokens
                        sample_masks.append(sample_mask)
                except Exception as e:
                    # Handle generation errors gracefully
                    print(f"Error during sample generation: {str(e)}")
                    # Use a fallback: just use the chosen sample directly
                    samples.append(chosen_ids[i:i+1])
                    sample_rewards.append(torch.tensor(0.0, device=self.device))
                    sample_masks.append(chosen_mask[i:i+1])
            
            # Check if we have enough valid samples
            if len(samples) < self.num_samples:
                print(f"Warning: Only generated {len(samples)}/{self.num_samples} valid samples")
                if len(samples) == 0:
                    # Skip this example if no valid samples
                    continue
            
            # Compute RLOO objective for each sample
            for j in range(len(samples)):
                # Compute baseline: average reward of other samples
                if len(samples) > 1:
                    other_rewards = [sample_rewards[k] for k in range(len(samples)) if k != j]
                    baseline = sum(other_rewards) / (len(samples) - 1)
                else:
                    # Fallback for single sample case
                    baseline = torch.tensor(0.0, device=self.device)
                
                # Compute advantage: R(y_i,x) - 1/(k-1) * sum_j!=i R(y_j,x)
                advantage = sample_rewards[j] - baseline
                
                # Clamp advantage to reasonable values to prevent numerical instability
                advantage = torch.clamp(advantage, min=-10.0, max=10.0)
                
                # Get policy logits for the sample
                try:
                    outputs = self.model(
                        input_ids=samples[j],
                        attention_mask=sample_masks[j]  # Use mask that zeros out prompt tokens
                    )
                    
                    # Compute log probabilities
                    logits = outputs.logits[..., :-1, :].contiguous()
                    labels = samples[j][..., 1:].contiguous()
                    sample_mask = sample_masks[j][..., 1:].contiguous()  # Shift mask for next-token prediction
                    
                    # Compute log probabilities for the target tokens
                    log_probs = F.log_softmax(logits, dim=-1)
                    target_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                    
                    # Apply attention mask to consider only response tokens
                    target_log_probs = target_log_probs * sample_mask
                    
                    # Compute mean log probability over non-zero elements only
                    non_zero_elements = sample_mask.sum()
                    epsilon = 1e-8  # Small value to avoid division by zero
                    if non_zero_elements > 0:
                        mean_log_prob = target_log_probs.sum() / max(non_zero_elements, epsilon)
                    else:
                        mean_log_prob = target_log_probs.mean()  # Fallback
                    
                    # Check for NaN values
                    if torch.isnan(mean_log_prob).any():
                        print(f"Warning: NaN detected in mean_log_prob for sample {j}")
                        continue
                    
                    # Update total loss: advantage * ∇log π(y_i|x)
                    total_loss = total_loss + advantage * -mean_log_prob  # Negative because we want to maximize
                    total_rewards.append(sample_rewards[j].item())
                    
                except Exception as e:
                    print(f"Error processing sample {j}: {str(e)}")
                    continue
        
        # Handle case where all samples failed
        if len(total_rewards) == 0:
            print("Warning: All samples failed processing. Using fallback loss value.")
            total_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
            avg_reward = 0.0
        else:
            # Average loss over valid samples
            total_loss = total_loss / len(total_rewards)
            avg_reward = sum(total_rewards) / len(total_rewards)
            
            # Final NaN check
            if torch.isnan(total_loss).any():
                print("Warning: NaN loss detected after averaging. Using fallback loss value.")
                total_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        # Scale the loss by the number of accumulation steps
        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        
        # Only perform optimization step after accumulating gradients
        if (self.current_step + 1) % self.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        self.current_step += 1
        
        return {
            "loss": total_loss.item() if not torch.isnan(total_loss).any().item() else 1.0,
            "avg_reward": avg_reward
        }

    def train(self, dataloader=None, num_epochs=1, max_steps=None):
        """
        Train the model on the provided dataloader.
        
        Args:
            dataloader: DataLoader containing training data
            num_epochs: Number of epochs to train for
            max_steps: Maximum number of steps to train for (overrides num_epochs if provided)
        
        Returns:
            Dictionary containing training metrics
        """
        if dataloader is None:
            raise ValueError("dataloader must be provided")
        
        self.model.train()
        self.reward_model.eval()
        
        device = next(self.model.parameters()).device
        step_count = 0
        total_loss = 0
        total_reward = 0
        total_steps = 0
        
        # Training loop
        from tqdm import tqdm
        
        if max_steps is not None:
            # If max_steps is provided, loop until reaching that many steps
            progress_bar = tqdm(total=max_steps, desc="Training")
            # Calculate true epoch based on steps and dataset size
            steps_per_epoch = len(dataloader)
            while step_count < max_steps:
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Get batch size
                    batch_size = batch["chosen_input_ids"].size(0)
                    
                    # Use the prompt_response_mask from the dataset instead of computing it
                    prompt_input_ids = batch["chosen_input_ids"].clone()
                    prompt_attention_mask = batch["chosen_attention_mask"].clone()
                    
                    # Use the masks from the dataset that properly distinguish prompt and response tokens
                    chosen_attention_mask = batch["chosen_attention_mask"].clone() * batch["chosen_prompt_response_mask"].clone()
                    rejected_attention_mask = batch["rejected_attention_mask"].clone() * batch["rejected_prompt_response_mask"].clone()
                    
                    if self.debug_mode and step_count % 50 == 0:
                        print("\n--- Prompt/Response Analysis ---")
                        print(f"Batch size: {batch_size}")
                        print(f"Using dataset-provided prompt/response masks")
                        
                        # Check mask shapes
                        print(f"Chosen mask shape: {chosen_attention_mask.shape}")
                        print(f"Rejected mask shape: {rejected_attention_mask.shape}")
                        
                        # Check correct masking for each example individually
                        mask_check_results = []
                        for b in range(min(batch["chosen_input_ids"].size(0), 3)):  # Check up to 3 examples
                            # Count non-zero elements in prompt and response regions
                            prompt_tokens = (batch["chosen_prompt_response_mask"][b] == 0).sum().item()
                            response_tokens = (batch["chosen_prompt_response_mask"][b] == 1).sum().item()
                            mask_check_results.append(f"Example {b}: prompt tokens={prompt_tokens}, response tokens={response_tokens}")
                        
                        print("Mask check (prompt=0, response=1):")
                        for result in mask_check_results:
                            print(f"  {result}")
                        print("-------------------------------")
                    
                    # Forward and backward pass
                    metrics = self.train_step(
                        prompt_ids=prompt_input_ids,
                        chosen_ids=batch["chosen_input_ids"],
                        rejected_ids=batch["rejected_input_ids"],
                        prompt_mask=prompt_attention_mask,
                        chosen_mask=chosen_attention_mask,
                        rejected_mask=rejected_attention_mask,
                        chosen_prompt_response_mask=batch["chosen_prompt_response_mask"],
                        rejected_prompt_response_mask=batch["rejected_prompt_response_mask"]
                    )
                    
                    total_loss += metrics["loss"]
                    total_reward += metrics.get("avg_reward", 0)
                    total_steps += 1
                    step_count += 1
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": metrics["loss"],
                        "avg_reward": metrics.get("avg_reward", 0)
                    })
                    
                    # Log epoch-level metrics if using wandb
                    if self.use_wandb:
                        # Calculate current epoch as a float to show partial progress
                        current_epoch = step_count / steps_per_epoch
                        wandb.log({
                            "epoch": current_epoch,
                            "epoch_avg_loss": total_loss / total_steps,
                            "epoch_avg_chosen_ratio": total_reward / total_steps,
                            "epoch_avg_rejected_ratio": total_reward / total_steps,
                            "learning_rate": self.optimizer.param_groups[0]["lr"]
                        }, step=self.current_step)
                    
                    if step_count >= max_steps:
                        break
                
        else:
            # Train for a fixed number of epochs
            for epoch in range(num_epochs):
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
                
                for batch in progress_bar:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Get batch size
                    batch_size = batch["chosen_input_ids"].size(0)
                    
                    # Use the prompt_response_mask from the dataset instead of computing it
                    prompt_input_ids = batch["chosen_input_ids"].clone()
                    prompt_attention_mask = batch["chosen_attention_mask"].clone()
                    
                    # Use the masks from the dataset that properly distinguish prompt and response tokens
                    chosen_attention_mask = batch["chosen_attention_mask"].clone() * batch["chosen_prompt_response_mask"].clone()
                    rejected_attention_mask = batch["rejected_attention_mask"].clone() * batch["rejected_prompt_response_mask"].clone()
                    
                    if self.debug_mode and step_count % 50 == 0:
                        print("\n--- Prompt/Response Analysis ---")
                        print(f"Batch size: {batch_size}")
                        print(f"Using dataset-provided prompt/response masks")
                        
                        # Check mask shapes
                        print(f"Chosen mask shape: {chosen_attention_mask.shape}")
                        print(f"Rejected mask shape: {rejected_attention_mask.shape}")
                        
                        # Check correct masking for each example individually
                        mask_check_results = []
                        for b in range(min(batch["chosen_input_ids"].size(0), 3)):  # Check up to 3 examples
                            # Count non-zero elements in prompt and response regions
                            prompt_tokens = (batch["chosen_prompt_response_mask"][b] == 0).sum().item()
                            response_tokens = (batch["chosen_prompt_response_mask"][b] == 1).sum().item()
                            mask_check_results.append(f"Example {b}: prompt tokens={prompt_tokens}, response tokens={response_tokens}")
                        
                        print("Mask check (prompt=0, response=1):")
                        for result in mask_check_results:
                            print(f"  {result}")
                        print("-------------------------------")
                    
                    # Forward and backward pass
                    metrics = self.train_step(
                        prompt_ids=prompt_input_ids,
                        chosen_ids=batch["chosen_input_ids"],
                        rejected_ids=batch["rejected_input_ids"],
                        prompt_mask=prompt_attention_mask,
                        chosen_mask=chosen_attention_mask,
                        rejected_mask=rejected_attention_mask,
                        chosen_prompt_response_mask=batch["chosen_prompt_response_mask"],
                        rejected_prompt_response_mask=batch["rejected_prompt_response_mask"]
                    )
                    
                    total_loss += metrics["loss"]
                    total_reward += metrics.get("avg_reward", 0)
                    total_steps += 1
                    step_count += 1
                    
                    progress_bar.set_postfix({
                        "loss": metrics["loss"],
                        "avg_reward": metrics.get("avg_reward", 0)
                    })
        
        # Return average metrics
        return {
            "loss": total_loss / total_steps,
            "avg_reward": total_reward / total_steps
        }
