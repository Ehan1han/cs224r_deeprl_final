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
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
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
        
        loss = compute_loss(outputs, labels, attention_mask)
        
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
        rejected_mask: torch.Tensor
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
        
        # Compute mean log probabilities over sequence length
        chosen_log_probs = (chosen_log_probs * chosen_mask).sum(dim=-1) / chosen_mask.sum(dim=-1)
        rejected_log_probs = (rejected_log_probs * rejected_mask).sum(dim=-1) / rejected_mask.sum(dim=-1)
        chosen_ref_log_probs = (chosen_ref_log_probs * chosen_mask).sum(dim=-1) / chosen_mask.sum(dim=-1)
        rejected_ref_log_probs = (rejected_ref_log_probs * rejected_mask).sum(dim=-1) / rejected_mask.sum(dim=-1)
        
        # Compute log probability ratios
        chosen_ratio = chosen_log_probs - chosen_ref_log_probs
        rejected_ratio = rejected_log_probs - rejected_ref_log_probs
        
        # Compute DPO loss: -E[log σ(β * (log πθ(yw|x)/πref(yw|x) - log πθ(yl|x)/πref(yl|x)))]
        loss = -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio)).mean()
        
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
        
        metrics = {
            "loss": loss.item(),
            "chosen_ratio": chosen_ratio.mean().item(),
            "rejected_ratio": rejected_ratio.mean().item(),
            "chosen_log_probs": chosen_log_probs.mean().item(),
            "rejected_log_probs": rejected_log_probs.mean().item(),
            "chosen_ref_log_probs": chosen_ref_log_probs.mean().item(),
            "rejected_ref_log_probs": rejected_ref_log_probs.mean().item()
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
                    
                    # Extract common prefix between chosen and rejected inputs (prompt)
                    batch_size = batch["chosen_input_ids"].size(0)
                    seq_length = batch["chosen_input_ids"].size(1)
                    
                    # Initialize with all ones (assuming everything is common)
                    common_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
                    
                    # Find where chosen and rejected differ (element-wise comparison)
                    split_points = []
                    for b in range(batch_size):
                        split_point = seq_length  # Default: whole sequence is prompt
                        for i in range(seq_length):
                            if batch["chosen_input_ids"][b, i] != batch["rejected_input_ids"][b, i]:
                                split_point = i
                                break
                        split_points.append(split_point)
                        common_mask[b, split_point:] = False
                    
                    # Create prompt_input_ids using only the common prefix (masked version)
                    prompt_input_ids = batch["chosen_input_ids"].clone() 
                    prompt_attention_mask = batch["chosen_attention_mask"].clone()
                    
                    # Zero out non-common parts (visualization only - masks handle this in computation)
                    prompt_input_ids = prompt_input_ids * common_mask.long()
                    
                    if self.debug_mode and step_count % 50 == 0:
                        print("\n--- Prompt/Response Analysis ---")
                        print(f"Batch size: {batch_size}")
                        print(f"Max sequence length: {seq_length}")
                        print(f"Split points: {split_points}")
                        print(f"Prompt shape: {prompt_input_ids.shape}")
                        print(f"Chosen shape: {batch['chosen_input_ids'].shape}")
                        
                        # Check correct masking for each example individually
                        mask_check_results = []
                        for b in range(min(batch_size, 3)):  # Check up to 3 examples
                            # Verify that prompt tokens have attention mask 0
                            prompt_zeros = (batch['chosen_attention_mask'][b, :split_points[b]] == 0).all().item()
                            # Verify that response tokens have attention mask 1
                            response_ones = (batch['chosen_attention_mask'][b, split_points[b]:] == 1).all().item()
                            mask_check_results.append(f"Example {b}: prompt zeros={prompt_zeros}, response ones={response_ones}")
                        
                        print("Attention mask check (should be 0=prompt, 1=response):")
                        for result in mask_check_results:
                            print(f"  {result}")
                        print("-------------------------------")
                    
                    # Remove the call to fix_dpo_masks - use original masks
                    chosen_mask = batch["chosen_attention_mask"]
                    rejected_mask = batch["rejected_attention_mask"]
                    
                    # Forward and backward pass
                    metrics = self.train_step(
                        prompt_ids=prompt_input_ids,
                        chosen_ids=batch["chosen_input_ids"],
                        rejected_ids=batch["rejected_input_ids"],
                        prompt_mask=prompt_attention_mask,
                        chosen_mask=chosen_mask,
                        rejected_mask=rejected_mask
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
                    
                    # Extract common prefix between chosen and rejected inputs (prompt)
                    batch_size = batch["chosen_input_ids"].size(0)
                    seq_length = batch["chosen_input_ids"].size(1)
                    
                    # Initialize with all ones (assuming everything is common)
                    common_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
                    
                    # Find where chosen and rejected differ (element-wise comparison)
                    split_points = []
                    for b in range(batch_size):
                        split_point = seq_length  # Default: whole sequence is prompt
                        for i in range(seq_length):
                            if batch["chosen_input_ids"][b, i] != batch["rejected_input_ids"][b, i]:
                                split_point = i
                                break
                        split_points.append(split_point)
                        common_mask[b, split_point:] = False
                    
                    # Create prompt_input_ids using only the common prefix (masked version)
                    prompt_input_ids = batch["chosen_input_ids"].clone() 
                    prompt_attention_mask = batch["chosen_attention_mask"].clone()
                    
                    # Zero out non-common parts (visualization only - masks handle this in computation)
                    prompt_input_ids = prompt_input_ids * common_mask.long()
                    
                    if self.debug_mode and step_count % 50 == 0:
                        print("\n--- Prompt/Response Analysis ---")
                        print(f"Batch size: {batch_size}")
                        print(f"Max sequence length: {seq_length}")
                        print(f"Split points: {split_points}")
                        print(f"Prompt shape: {prompt_input_ids.shape}")
                        print(f"Chosen shape: {batch['chosen_input_ids'].shape}")
                        
                        # Check correct masking for each example individually
                        mask_check_results = []
                        for b in range(min(batch_size, 3)):  # Check up to 3 examples
                            # Verify that prompt tokens have attention mask 0
                            prompt_zeros = (batch['chosen_attention_mask'][b, :split_points[b]] == 0).all().item()
                            # Verify that response tokens have attention mask 1
                            response_ones = (batch['chosen_attention_mask'][b, split_points[b]:] == 1).all().item()
                            mask_check_results.append(f"Example {b}: prompt zeros={prompt_zeros}, response ones={response_ones}")
                        
                        print("Attention mask check (should be 0=prompt, 1=response):")
                        for result in mask_check_results:
                            print(f"  {result}")
                        print("-------------------------------")
                    
                    # Remove the call to fix_dpo_masks - use original masks
                    chosen_mask = batch["chosen_attention_mask"]
                    rejected_mask = batch["rejected_attention_mask"]
                    
                    # Forward and backward pass
                    metrics = self.train_step(
                        prompt_ids=prompt_input_ids,
                        chosen_ids=batch["chosen_input_ids"],
                        rejected_ids=batch["rejected_input_ids"],
                        prompt_mask=prompt_attention_mask,
                        chosen_mask=chosen_mask,
                        rejected_mask=rejected_mask
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
        rejected_mask: torch.Tensor
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
        total_loss = 0
        total_rewards = []
        
        # Process each example in the batch
        for i in range(batch_size):
            # Generate k samples for the current input
            samples = []
            sample_rewards = []
            
            # Generate k samples
            for _ in range(self.num_samples):
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
            
            # Compute RLOO objective for each sample
            for j in range(self.num_samples):
                # Compute baseline: average reward of other samples
                other_rewards = [sample_rewards[k] for k in range(self.num_samples) if k != j]
                baseline = sum(other_rewards) / (self.num_samples - 1)
                
                # Compute advantage: R(y_i,x) - 1/(k-1) * sum_j!=i R(y_j,x)
                advantage = sample_rewards[j] - baseline
                
                # Get policy logits for the sample
                outputs = self.model(
                    input_ids=samples[j],
                    attention_mask=torch.ones_like(samples[j])
                )
                
                # Compute log probabilities
                logits = outputs.logits[..., :-1, :].contiguous()
                labels = samples[j][..., 1:].contiguous()
                
                # Compute log probabilities for the target tokens
                log_probs = F.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                
                # Compute mean log probability
                mean_log_prob = target_log_probs.mean()
                
                # Update total loss: advantage * ∇log π(y_i|x)
                total_loss += advantage * mean_log_prob
                total_rewards.append(sample_rewards[j].item())
        
        # Average loss over batch and samples
        total_loss = -total_loss / (batch_size * self.num_samples)  # Negative because we want to maximize
        avg_reward = sum(total_rewards) / len(total_rewards)
        
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
            "loss": total_loss.item(),
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
                    
                    # Extract common prefix between chosen and rejected inputs (prompt)
                    batch_size = batch["chosen_input_ids"].size(0)
                    seq_length = batch["chosen_input_ids"].size(1)
                    
                    # Initialize with all ones (assuming everything is common)
                    common_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
                    
                    # Find where chosen and rejected differ (element-wise comparison)
                    split_points = []
                    for b in range(batch_size):
                        split_point = seq_length  # Default: whole sequence is prompt
                        for i in range(seq_length):
                            if batch["chosen_input_ids"][b, i] != batch["rejected_input_ids"][b, i]:
                                split_point = i
                                break
                        split_points.append(split_point)
                        common_mask[b, split_point:] = False
                    
                    # Create prompt_input_ids using only the common prefix (masked version)
                    prompt_input_ids = batch["chosen_input_ids"].clone() 
                    prompt_attention_mask = batch["chosen_attention_mask"].clone()
                    
                    # Zero out non-common parts (visualization only - masks handle this in computation)
                    prompt_input_ids = prompt_input_ids * common_mask.long()
                    
                    if self.debug_mode and step_count % 50 == 0:
                        print("\n--- Prompt/Response Analysis ---")
                        print(f"Batch size: {batch_size}")
                        print(f"Max sequence length: {seq_length}")
                        print(f"Split points: {split_points}")
                        print(f"Prompt shape: {prompt_input_ids.shape}")
                        print(f"Chosen shape: {batch['chosen_input_ids'].shape}")
                        
                        # Check correct masking for each example individually
                        mask_check_results = []
                        for b in range(min(batch_size, 3)):  # Check up to 3 examples
                            # Verify that prompt tokens have attention mask 0
                            prompt_zeros = (batch['chosen_attention_mask'][b, :split_points[b]] == 0).all().item()
                            # Verify that response tokens have attention mask 1
                            response_ones = (batch['chosen_attention_mask'][b, split_points[b]:] == 1).all().item()
                            mask_check_results.append(f"Example {b}: prompt zeros={prompt_zeros}, response ones={response_ones}")
                        
                        print("Attention mask check (should be 0=prompt, 1=response):")
                        for result in mask_check_results:
                            print(f"  {result}")
                        print("-------------------------------")
                    
                    # Remove the call to fix_dpo_masks - use original masks
                    chosen_mask = batch["chosen_attention_mask"]
                    rejected_mask = batch["rejected_attention_mask"]
                    
                    # Forward and backward pass
                    metrics = self.train_step(
                        prompt_ids=prompt_input_ids,
                        chosen_ids=batch["chosen_input_ids"],
                        rejected_ids=batch["rejected_input_ids"],
                        prompt_mask=prompt_attention_mask,
                        chosen_mask=chosen_mask,
                        rejected_mask=rejected_mask
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
                    
                    # Extract common prefix between chosen and rejected inputs (prompt)
                    batch_size = batch["chosen_input_ids"].size(0)
                    seq_length = batch["chosen_input_ids"].size(1)
                    
                    # Initialize with all ones (assuming everything is common)
                    common_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
                    
                    # Find where chosen and rejected differ (element-wise comparison)
                    split_points = []
                    for b in range(batch_size):
                        split_point = seq_length  # Default: whole sequence is prompt
                        for i in range(seq_length):
                            if batch["chosen_input_ids"][b, i] != batch["rejected_input_ids"][b, i]:
                                split_point = i
                                break
                        split_points.append(split_point)
                        common_mask[b, split_point:] = False
                    
                    # Create prompt_input_ids using only the common prefix (masked version)
                    prompt_input_ids = batch["chosen_input_ids"].clone() 
                    prompt_attention_mask = batch["chosen_attention_mask"].clone()
                    
                    # Zero out non-common parts (visualization only - masks handle this in computation)
                    prompt_input_ids = prompt_input_ids * common_mask.long()
                    
                    if self.debug_mode and step_count % 50 == 0:
                        print("\n--- Prompt/Response Analysis ---")
                        print(f"Batch size: {batch_size}")
                        print(f"Max sequence length: {seq_length}")
                        print(f"Split points: {split_points}")
                        print(f"Prompt shape: {prompt_input_ids.shape}")
                        print(f"Chosen shape: {batch['chosen_input_ids'].shape}")
                        
                        # Check correct masking for each example individually
                        mask_check_results = []
                        for b in range(min(batch_size, 3)):  # Check up to 3 examples
                            # Verify that prompt tokens have attention mask 0
                            prompt_zeros = (batch['chosen_attention_mask'][b, :split_points[b]] == 0).all().item()
                            # Verify that response tokens have attention mask 1
                            response_ones = (batch['chosen_attention_mask'][b, split_points[b]:] == 1).all().item()
                            mask_check_results.append(f"Example {b}: prompt zeros={prompt_zeros}, response ones={response_ones}")
                        
                        print("Attention mask check (should be 0=prompt, 1=response):")
                        for result in mask_check_results:
                            print(f"  {result}")
                        print("-------------------------------")
                    
                    # Remove the call to fix_dpo_masks - use original masks
                    chosen_mask = batch["chosen_attention_mask"]
                    rejected_mask = batch["rejected_attention_mask"]
                    
                    # Forward and backward pass
                    metrics = self.train_step(
                        prompt_ids=prompt_input_ids,
                        chosen_ids=batch["chosen_input_ids"],
                        rejected_ids=batch["rejected_input_ids"],
                        prompt_mask=prompt_attention_mask,
                        chosen_mask=chosen_mask,
                        rejected_mask=rejected_mask
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
