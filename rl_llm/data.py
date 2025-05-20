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
            self.dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=actual_split)
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
        
        # Create attention mask that masks out query tokens
        if self.dataset_name == "smoltalk":
            # For SmolTalk, mask out all but the last assistant response
            messages = text.split(self.tokenizer.eos_token)
            if len(messages) > 1:
                last_assistant_response = messages[-1]
                response_tokens = self.tokenizer.encode(
                    last_assistant_response,
                    add_special_tokens=False
                )
                # Truncate from the right (completion side)
                if len(response_tokens) > self.max_length:
                    response_tokens = response_tokens[-self.max_length:]
                encodings["attention_mask"][0, :-len(response_tokens)] = 0
        elif self.dataset_name == "ultrafeedback":
            # For UltraFeedback, mask out the chosen response
            split_text = text.split(self.tokenizer.eos_token)
            if len(split_text) > 1:
                messages = [
                    {"role": "user", "content": split_text[0]},
                    {"role": "assistant", "content": split_text[1]}
                ]
                chosen_tokens = self.tokenizer.encode(
                    self.tokenizer.apply_chat_template(
                        [messages[1]],
                        tokenize=False,
                        add_generation_prompt=False
                    ),
                    add_special_tokens=False
                )
                encodings["attention_mask"][0, :-len(chosen_tokens)] = 0
        
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
