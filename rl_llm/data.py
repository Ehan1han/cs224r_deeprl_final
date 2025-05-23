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
        subset_size: Optional[int] = None,
        prompt_max_length_ratio: float = 0.7,  # New param: maximum ratio of prompt to max_length
        verbose: bool = True  # New param: verbose logging
    ):
        """
        Initialize the preference dataset.
        
        Args:
            dataset_name: Either "smoltalk" (maps to "HuggingFaceTB/smol-smoltalk") or "ultrafeedback"
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            split: Dataset split to use
            subset_size: Optional subset size for testing
            prompt_max_length_ratio: Maximum ratio of prompt length to max_length for filtering (DPO only)
            verbose: Whether to print verbose information during loading
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.verbose = verbose
        
        # Load dataset
        if dataset_name == "ultrafeedback":
            # Map generic split names to UltraFeedback specific splits
            split_map = {
                "train": "train_prefs",
                "test": "test_prefs"
            }
            actual_split = split_map.get(split, split)
            self.dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=actual_split)
            
            # Filter out examples with too long prompts for DPO training
            if prompt_max_length_ratio > 0:
                prompt_max_tokens = int(max_length * prompt_max_length_ratio)
                
                # Count original dataset size
                original_size = len(self.dataset)
                
                # Function to check if prompt length is acceptable
                def is_acceptable_length(example):
                    # Format just the prompt to check its length
                    prompt_messages = [{"role": "user", "content": example["prompt"]}]
                    prompt_text = self._format_conversation(prompt_messages)
                    prompt_tokens = self.tokenizer.encode(prompt_text)
                    return len(prompt_tokens) <= prompt_max_tokens
                
                # Apply the filter
                self.dataset = self.dataset.filter(is_acceptable_length)
                
                # Report filtering results if verbose
                if verbose:
                    filtered_size = len(self.dataset)
                    removed = original_size - filtered_size
                    removed_percent = removed / original_size * 100 if original_size > 0 else 0
                    print(f"Filtered UltraFeedback dataset: removed {removed}/{original_size} examples ({removed_percent:.2f}%) with prompts longer than {prompt_max_tokens} tokens ({prompt_max_length_ratio*100:.0f}% of max_length)")
            
            # Apply subset selection after filtering
            if subset_size is not None:
                self.dataset = self.dataset.select(range(min(subset_size, len(self.dataset))))
                
        elif dataset_name == "smoltalk":
            self.dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
            
            # For SmolTalk dataset:
            # 1. Filter for length-2 conversations (1 user + 1 assistant message)
            self.dataset = self.dataset.filter(
                lambda x: len(x["messages"]) == 2 and 
                x["messages"][0]["role"] == "user" and 
                x["messages"][1]["role"] == "assistant"
            )
            
            # 2. Calculate 95th percentile of token lengths using a sample
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
                if verbose:
                    print(f"SmolTalk dataset: Setting max_length to 95th percentile: {self.max_length} (calculated from {sample_size} samples)")
            
            if subset_size is not None:
                self.dataset = self.dataset.select(range(min(subset_size, len(self.dataset))))
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.dataset_name = dataset_name.lower()

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation using the tokenizer's chat template if appropriate, otherwise join content as string."""
        def normalize_content(content):
            if isinstance(content, list):
                # If list of dicts, extract their 'content' fields
                if all(isinstance(x, dict) and 'content' in x for x in content):
                    return "\n".join(normalize_content(x['content']) for x in content)
                # If list of strings
                return "\n".join(str(x) for x in content)
            elif isinstance(content, dict) and 'content' in content:
                return normalize_content(content['content'])
            else:
                return str(content)

        # If messages is a list of dicts and tokenizer has a chat template, use it
        if (
            isinstance(messages, list)
            and all(isinstance(m, dict) and "role" in m and "content" in m for m in messages)
            and hasattr(self.tokenizer, 'chat_template')
            and self.tokenizer.chat_template is not None
        ):
            # Ensure all content fields are strings
            for m in messages:
                m["content"] = normalize_content(m["content"])
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback: join content if it's a list, or use as is
            if isinstance(messages, list):
                return "\n".join(normalize_content(m["content"]) if isinstance(m, dict) and "content" in m else normalize_content(m) for m in messages)
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
        
        # Initialize attention mask with zeros (all tokens masked out)
        # Important: We manually override the attention mask here, so we explicitly set it to zeros
        encodings["attention_mask"] = torch.zeros_like(encodings["input_ids"], dtype=torch.long)
        
        # First try to find the assistant marker
        assistant_start_marker = "<|im_start|>"
        assistant_role = "assistant"
        
        # Look for <|im_start|>assistant pattern
        assistant_start_pos = text.rfind(assistant_start_marker + assistant_role)
        
        # If not found as a combined string, try finding them separately
        if assistant_start_pos == -1:
            assistant_start_pos = text.rfind(assistant_start_marker)
            # Check if after <|im_start|> there's "assistant" within reasonable distance
            if assistant_start_pos != -1:
                # Look for "assistant" after <|im_start|> within a small window
                window_size = 20  # characters to look ahead
                window_end = min(assistant_start_pos + len(assistant_start_marker) + window_size, len(text))
                window_text = text[assistant_start_pos + len(assistant_start_marker):window_end]
                if assistant_role in window_text:
                    # Adjust position to include both markers
                    role_pos = window_text.find(assistant_role)
                    assistant_start_pos = assistant_start_pos + role_pos + len(assistant_start_marker)
        
        if assistant_start_pos != -1:
            # Get assistant response text (skip the markers and get the actual content)
            marker_end_pos = text.find("\n", assistant_start_pos)
            if marker_end_pos == -1:  # If no newline, look for space
                marker_end_pos = text.find(" ", assistant_start_pos)
            
            if marker_end_pos != -1:
                # Get text after the marker and newline/space
                assistant_content = text[marker_end_pos+1:]
            else:
                # Fallback to using everything after the marker
                assistant_content = text[assistant_start_pos:]
            
            # Encode just the assistant content part
            assistant_tokens = self.tokenizer.encode(
                assistant_content,
                add_special_tokens=False
            )
            
            # Encode the full text up to the assistant content to find token position
            prefix_text = text[:marker_end_pos+1] if marker_end_pos != -1 else text[:assistant_start_pos]
            prefix_tokens = self.tokenizer.encode(
                prefix_text,
                add_special_tokens=False
            )
            
            # Set attention mask: 0 for prompt, 1 for response
            seq_length = encodings["input_ids"].size(1)
            
            # Calculate position where assistant content starts (after prefix)
            # We need to find this position in the tokenized sequence
            # Starting position is the length of the prefix tokens
            start_pos = min(len(prefix_tokens), seq_length)
            
            # Set all tokens after this position to have attention mask = 1
            encodings["attention_mask"][0, start_pos:] = 1
            
            # Verify we have a reasonable number of 1s
            ones_count = encodings["attention_mask"].sum().item()
            if ones_count < 5:
                # If too few 1s (edge case), use a fallback: set last 30% to 1
                fallback_boundary = int(seq_length * 0.7)
                encodings["attention_mask"][0, :fallback_boundary] = 0
                encodings["attention_mask"][0, fallback_boundary:] = 1
        else:
            # Alternative approach: find where tokens change between prompt and response
            # For example, try to find common markers like:
            alt_markers = ["<assistant>", "Assistant:", "\nassistant:", "\nAssistant:", "AI:", "\nAI:"]
            found = False
            
            for marker in alt_markers:
                marker_pos = text.rfind(marker)
                if marker_pos != -1:
                    # Found an alternative marker
                    marker_end_pos = text.find("\n", marker_pos)
                    if marker_end_pos == -1:
                        marker_end_pos = text.find(" ", marker_pos)
                    
                    if marker_end_pos != -1:
                        assistant_content = text[marker_end_pos+1:]
                    else:
                        assistant_content = text[marker_pos:]
                    
                    # Encode the assistant content
                    assistant_tokens = self.tokenizer.encode(
                        assistant_content,
                        add_special_tokens=False
                    )
                    
                    # Encode prefix to find position
                    prefix_text = text[:marker_end_pos+1] if marker_end_pos != -1 else text[:marker_pos]
                    prefix_tokens = self.tokenizer.encode(
                        prefix_text,
                        add_special_tokens=False
                    )
                    
                    # Set tokens after prefix to have attention mask = 1
                    seq_length = encodings["input_ids"].size(1)
                    start_pos = min(len(prefix_tokens), seq_length)
                    encodings["attention_mask"][0, start_pos:] = 1
                    
                    found = True
                    break
            
            if not found:
                # Fallback if no marker found (shouldn't happen with proper data)
                # Set the latter half to 1 (response) and first half to 0 (prompt)
                seq_length = encodings["input_ids"].size(1)
                boundary = int(seq_length * 0.5)  # Default to middle of sequence
                encodings["attention_mask"][0, :boundary] = 0
                encodings["attention_mask"][0, boundary:] = 1
        
        return encodings

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        if self.dataset_name == "smoltalk":
            # Format SmolTalk conversation
            text = self._format_conversation(item["messages"])
            encodings = self._tokenize(text)
            
            return {
                "input_ids": encodings["input_ids"].squeeze(),
                "attention_mask": encodings["attention_mask"].squeeze(),
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
            
            chosen_text = self._format_conversation(messages_chosen)
            rejected_text = self._format_conversation(messages_rejected)
            
            chosen_encodings = self._tokenize(chosen_text)
            rejected_encodings = self._tokenize(rejected_text)
            
            return {
                "chosen_input_ids": chosen_encodings["input_ids"].squeeze(),
                "chosen_attention_mask": chosen_encodings["attention_mask"].squeeze(),
                "rejected_input_ids": rejected_encodings["input_ids"].squeeze(),
                "rejected_attention_mask": rejected_encodings["attention_mask"].squeeze()
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

def fix_dpo_masks(batch):
    """
    Fix attention masks for DPO training to ensure they align with content boundaries.
    
    For proper DPO training:
    - Prompt tokens should have attention_mask=0
    - Response tokens should have attention_mask=1
    
    Args:
        batch: Dictionary containing chosen_input_ids, rejected_input_ids,
               chosen_attention_mask, rejected_attention_mask
               
    Returns:
        Dictionary with fixed attention masks
    """
    # Make a copy of the batch to avoid modifying the original
    fixed_batch = {
        "chosen_input_ids": batch["chosen_input_ids"],
        "rejected_input_ids": batch["rejected_input_ids"],
        "chosen_attention_mask": batch["chosen_attention_mask"].clone(),
        "rejected_attention_mask": batch["rejected_attention_mask"].clone()
    }
    
    batch_size = batch["chosen_input_ids"].size(0)
    
    for b in range(batch_size):
        # Find the split point (where chosen and rejected first differ)
        chosen_ids = batch["chosen_input_ids"][b].tolist()
        rejected_ids = batch["rejected_input_ids"][b].tolist()
        
        # Default: split at the end (all prompt)
        split_point = len(chosen_ids)
        
        # Find first difference
        for i in range(min(len(chosen_ids), len(rejected_ids))):
            if chosen_ids[i] != rejected_ids[i]:
                split_point = i
                break
        
        # Fix chosen attention mask: 0s before split point, 1s after
        fixed_batch["chosen_attention_mask"][b, :split_point] = 0
        fixed_batch["chosen_attention_mask"][b, split_point:] = 1
        
        # Fix rejected attention mask: 0s before split point, 1s after
        fixed_batch["rejected_attention_mask"][b, :split_point] = 0
        fixed_batch["rejected_attention_mask"][b, split_point:] = 1
    
    return fixed_batch
