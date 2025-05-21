from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset
import numpy as np

class PreferenceDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "train",
        subset_size: Optional[int] = None
    ):
        """
        Initialize the preference dataset.
        
        Args:
            dataset_name: Either "smoltalk" (maps to "HuggingFaceTB/smol-smoltalk") or "ultrafeedback"
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            split: Dataset split to use
            subset_size: Optional subset size for testing
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        if dataset_name == "ultrafeedback":
            # Map generic split names to UltraFeedback specific splits
            split_map = {
                "train": "train_prefs",
                "test": "test_prefs"
            }
            actual_split = split_map.get(split, split)
            raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=actual_split)
            
            # Filter out examples with prompts that exceed max_length
            filtered_indices = []
            skipped_count = 0
            
            print(f"Filtering UltraFeedback dataset to remove examples with long prompts...")
            for i, item in enumerate(raw_dataset):
                try:
                    # Try to estimate prompt length by tokenizing it
                    prompt_tokens = len(tokenizer.encode(item["prompt"]))
                    if prompt_tokens <= max_length * 0.7:  # Leave 30% space for responses
                        filtered_indices.append(i)
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"Error processing example {i}: {str(e)}")
                    skipped_count += 1
            
            if skipped_count > 0:
                print(f"Filtered out {skipped_count}/{len(raw_dataset)} examples with prompts exceeding {max_length * 0.7} tokens")
            
            # If subset_size is provided, take a subset of the filtered indices
            if subset_size is not None:
                filtered_indices = filtered_indices[:min(subset_size, len(filtered_indices))]
                
            self.dataset = raw_dataset.select(filtered_indices)
            print(f"Final dataset size: {len(self.dataset)} examples")
            
        elif dataset_name == "smoltalk":
            raw_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
            
            # For SmolTalk dataset:
            # 1. Filter for length-2 conversations (1 user + 1 assistant message)
            raw_dataset = raw_dataset.filter(
                lambda x: len(x["messages"]) == 2 and 
                x["messages"][0]["role"] == "user" and 
                x["messages"][1]["role"] == "assistant"
            )
            
            # Filter out examples with prompts that exceed max_length
            filtered_indices = []
            skipped_count = 0
            
            print(f"Filtering SmolTalk dataset to remove examples with long prompts...")
            for i, item in enumerate(raw_dataset):
                try:
                    # Try to estimate prompt length by tokenizing the user message
                    prompt_tokens = len(tokenizer.encode(item["messages"][0]["content"]))
                    if prompt_tokens <= max_length * 0.7:  # Leave 30% space for responses
                        filtered_indices.append(i)
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"Error processing example {i}: {str(e)}")
                    skipped_count += 1
            
            if skipped_count > 0:
                print(f"Filtered out {skipped_count}/{len(raw_dataset)} examples with prompts exceeding {max_length * 0.7} tokens")
            
            # If subset_size is provided, take a subset of the filtered indices
            if subset_size is not None:
                filtered_indices = filtered_indices[:min(subset_size, len(filtered_indices))]
                
            self.dataset = raw_dataset.select(filtered_indices)
            print(f"Final dataset size: {len(self.dataset)} examples")
            
            # Calculate 95th percentile of token lengths using a sample from filtered dataset
            if len(self.dataset) > 0:
                # Use a sample of 1000 conversations for faster processing
                sample_size = min(1000, len(self.dataset))
                sample_indices = np.random.choice(len(self.dataset), sample_size, replace=False)
                token_lengths = []
                
                for idx in sample_indices:
                    item = self.dataset[int(idx)]
                    text = self._format_conversation(item["messages"])
                    length = len(self.tokenizer.encode(text))
                    token_lengths.append(length)
                
                # Set max_length to 95th percentile
                self.max_length = int(np.percentile(token_lengths, 95))
                print(f"SmolTalk dataset: Setting max_length to 95th percentile: {self.max_length} (calculated from {sample_size} samples)")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.dataset_name = dataset_name.lower()

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation by simply joining content as string."""
        def normalize_content(content):
            """Ensure content is normalized to a string regardless of input type."""
            if content is None:
                return ""
            elif isinstance(content, list):
                # If list of dicts, extract their 'content' fields
                if all(isinstance(x, dict) and 'content' in x for x in content):
                    return "\n".join(normalize_content(x['content']) for x in content)
                # If list of strings or mixed types
                return "\n".join(normalize_content(x) for x in content)
            elif isinstance(content, dict):
                # Handle dictionaries: if it has 'content', use that, otherwise stringify the dict
                if 'content' in content:
                    return normalize_content(content['content'])
                else:
                    # Convert dict to string if no 'content' field
                    return str(content)
            else:
                # For any other type, convert to string
                return str(content)

        # Simply join content
        if isinstance(messages, list):
            return "\n".join(normalize_content(m.get("content", m)) if isinstance(m, dict) else normalize_content(m) for m in messages)
        return normalize_content(messages)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text and create attention mask."""
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encodings
    
    def _tokenize_with_role_masks(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize text and create masks that distinguish between prompt (user) and response (assistant).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' fields
            
        Returns:
            Dictionary with input_ids, attention_mask, and prompt_response_mask tensors
            prompt_response_mask: 0 for prompt tokens (user), 1 for response tokens (assistant)
        """
        # First, tokenize each message separately to track token counts
        message_tokens = []
        for message in messages:
            # Normalize content to ensure it's a string
            content = message["content"]
            if not isinstance(content, str):
                content = self._format_conversation([{"content": content}])
                
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            message_tokens.append(tokens)
        
        # Create the full text for tokenization (as we did before)
        full_text = self._format_conversation(messages)
        
        # Get the full tokenization with padding
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create a mask where 0 = prompt tokens (user), 1 = response tokens (assistant)
        prompt_response_mask = torch.zeros_like(encodings["input_ids"])
        
        # Process the tokenized result to create the role mask
        current_pos = 0
        for i, message in enumerate(messages):
            token_count = len(message_tokens[i])
            role = message["role"]
            
            # Mark positions for this message
            if role == "assistant":
                # Set 1 for assistant tokens (response)
                end_pos = min(current_pos + token_count, self.max_length - 1)
                prompt_response_mask[0, current_pos:end_pos] = 1
            # For user tokens, keep as 0 (prompt)
            
            current_pos += token_count
            if current_pos >= self.max_length - 1:
                break
        
        # Add the mask to the encodings
        encodings["prompt_response_mask"] = prompt_response_mask
        return encodings

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        if self.dataset_name == "smoltalk":
            # Format and tokenize SmolTalk conversation with role-based masks
            encodings = self._tokenize_with_role_masks(item["messages"])
            
            return {
                "input_ids": encodings["input_ids"].squeeze(),
                "attention_mask": encodings["attention_mask"].squeeze(),
                "prompt_response_mask": encodings["prompt_response_mask"].squeeze(),
                "labels": encodings["input_ids"].squeeze()
            }
            
        elif self.dataset_name == "ultrafeedback":
            # Format UltraFeedback conversation with preferred and dispreferred responses
            messages_chosen = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["chosen"]}
            ]
            messages_rejected = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["rejected"]}
            ]
            
            # Tokenize with role-based masks
            chosen_encodings = self._tokenize_with_role_masks(messages_chosen)
            rejected_encodings = self._tokenize_with_role_masks(messages_rejected)
            
            return {
                "chosen_input_ids": chosen_encodings["input_ids"].squeeze(),
                "chosen_attention_mask": chosen_encodings["attention_mask"].squeeze(),
                "chosen_prompt_response_mask": chosen_encodings["prompt_response_mask"].squeeze(),
                "rejected_input_ids": rejected_encodings["input_ids"].squeeze(),
                "rejected_attention_mask": rejected_encodings["attention_mask"].squeeze(),
                "rejected_prompt_response_mask": rejected_encodings["prompt_response_mask"].squeeze()
            }

def create_dataloader(
    dataset: PreferenceDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0  # Changed to 0 to avoid CUDA initialization issues
) -> DataLoader:
    """
    Create a PyTorch DataLoader for the preference dataset.
    
    Args:
        dataset: PreferenceDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading (set to 0 to avoid CUDA issues)
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
