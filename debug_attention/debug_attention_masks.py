import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
from rl_llm.data import PreferenceDataset, fix_dpo_masks
from rl_llm.evaluation import ModelEvaluator, load_eval_prompts

def visualize_attention_mask(input_ids, attention_mask, tokenizer, title):
    """Visualize token IDs and attention mask"""
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    
    # Create visualization
    result = [f"=== {title} ==="]
    result.append("Token | Attention | Text")
    result.append("-" * 50)
    
    for i, (token_id, mask_val, token_text) in enumerate(zip(input_ids.tolist(), 
                                                           attention_mask.tolist(),
                                                           tokens)):
        if token_id == tokenizer.pad_token_id:
            continue  # Skip padding tokens
        
        # Get special token representation
        if token_text.startswith("<") and token_text.endswith(">"):
            token_display = token_text
        else:
            # Display token text with spaces for clarity
            token_display = token_text.replace("▁", " ")
        
        result.append(f"{i:4d} | {mask_val:9d} | {token_display}")
    
    return "\n".join(result)

def find_assistant_boundary(text):
    """Find the boundary where assistant response starts"""
    # Try to find <|im_start|>assistant pattern
    assistant_start_marker = "<|im_start|>"
    assistant_role = "assistant"
    
    # Look for <|im_start|>assistant pattern first
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
                return assistant_start_pos
    
    if assistant_start_pos != -1:
        return assistant_start_pos
    
    # Alternative approach: find where tokens change between prompt and response
    alt_markers = ["<assistant>", "Assistant:", "\nassistant:", "\nAssistant:", "AI:", "\nAI:"]
    
    for marker in alt_markers:
        marker_pos = text.rfind(marker)
        if marker_pos != -1:
            return marker_pos
    
    # If still no marker found, use a heuristic
    return len(text) // 2  # Middle of text as fallback

def check_mask_alignment(input_ids, attention_mask, tokenizer, text):
    """Check if attention mask aligns with content boundaries"""
    # Find assistant boundary in text
    boundary_pos = find_assistant_boundary(text)
    
    # Get mask values
    mask = attention_mask.tolist()
    
    # Find first 1 in mask
    first_one_idx = mask.index(1) if 1 in mask else None
    
    # Calculate percentage of mask that is 1
    percent_ones = sum(mask) / len(mask) * 100
    
    # Check if there's a clear boundary (0s followed by 1s)
    has_clear_boundary = first_one_idx is not None and first_one_idx > 0 and first_one_idx < len(mask) - 1
    
    return {
        "text_boundary_pos": boundary_pos,
        "first_one_idx": first_one_idx,
        "has_clear_boundary": has_clear_boundary,
        "percent_ones": percent_ones
    }

def check_dpo_mask_fixes(batch):
    """Check if the DPO mask fixes are working correctly"""
    # Make a copy of the batch to compare before and after
    original_batch = {
        "chosen_input_ids": batch["chosen_input_ids"].clone(),
        "rejected_input_ids": batch["rejected_input_ids"].clone(),
        "chosen_attention_mask": batch["chosen_attention_mask"].clone(),
        "rejected_attention_mask": batch["rejected_attention_mask"].clone()
    }
    
    # Apply the fix_dpo_masks function
    fixed_batch = fix_dpo_masks(batch)
    
    # Check differences
    chosen_diff = (original_batch["chosen_attention_mask"] != fixed_batch["chosen_attention_mask"]).sum().item()
    rejected_diff = (original_batch["rejected_attention_mask"] != fixed_batch["rejected_attention_mask"]).sum().item()
    
    # Find first 1 in each mask
    orig_chosen_first_one = original_batch["chosen_attention_mask"].squeeze().tolist().index(1) if 1 in original_batch["chosen_attention_mask"].squeeze().tolist() else None
    fixed_chosen_first_one = fixed_batch["chosen_attention_mask"].squeeze().tolist().index(1) if 1 in fixed_batch["chosen_attention_mask"].squeeze().tolist() else None
    
    orig_rejected_first_one = original_batch["rejected_attention_mask"].squeeze().tolist().index(1) if 1 in original_batch["rejected_attention_mask"].squeeze().tolist() else None
    fixed_rejected_first_one = fixed_batch["rejected_attention_mask"].squeeze().tolist().index(1) if 1 in fixed_batch["rejected_attention_mask"].squeeze().tolist() else None
    
    # Determine where chosen and rejected start to differ
    chosen_ids = original_batch["chosen_input_ids"].squeeze().tolist()
    rejected_ids = original_batch["rejected_input_ids"].squeeze().tolist()
    
    # Find split point
    split_point = None
    for i in range(min(len(chosen_ids), len(rejected_ids))):
        if chosen_ids[i] != rejected_ids[i]:
            split_point = i
            break
    
    return {
        "chosen_mask_changes": chosen_diff,
        "rejected_mask_changes": rejected_diff,
        "orig_chosen_first_one": orig_chosen_first_one,
        "fixed_chosen_first_one": fixed_chosen_first_one,
        "orig_rejected_first_one": orig_rejected_first_one,
        "fixed_rejected_first_one": fixed_rejected_first_one,
        "split_point": split_point,
        "chosen_properly_aligned": fixed_chosen_first_one == split_point if split_point is not None and fixed_chosen_first_one is not None else False,
        "rejected_properly_aligned": fixed_rejected_first_one == split_point if split_point is not None and fixed_rejected_first_one is not None else False
    }

def debug_attention_masks(output_file="pipeline_debug_output.txt"):
    """Debug training and evaluation pipelines with all intermediate steps"""
    
    # Create/open output file
    with open(output_file, "w") as f:
        f.write("# Training and Evaluation Pipeline Debugging\n\n")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        #############################################################
        # PART 1: TRAINING PIPELINE DEBUGGING
        #############################################################
        f.write("## 1. TRAINING PIPELINE DEBUGGING\n\n")
        
        # 1.1 Debug SmolTalk dataset (SFT training)
        f.write("### 1.1 SmolTalk Dataset (SFT Training)\n\n")
        
        smoltalk_dataset = PreferenceDataset(
            dataset_name="smoltalk",
            tokenizer=tokenizer,
            max_length=512,
            subset_size=3  # Small sample for debugging
        )
        
        # Get examples from SmolTalk
        for idx in range(min(3, len(smoltalk_dataset))):
            # Get original data item
            item = smoltalk_dataset.dataset[idx]
            
            # Get the raw conversation data
            f.write(f"#### Example {idx+1}\n")
            f.write("**1. Original Data:**\n```json\n")
            f.write(str(item))
            f.write("\n```\n\n")
            
            # Format text for display (after _format_conversation)
            formatted_text = smoltalk_dataset._format_conversation(item["messages"])
            f.write("**2. After _format_conversation:**\n```\n")
            f.write(formatted_text)
            f.write("\n```\n\n")
            
            # Get tokenized sample
            sample = smoltalk_dataset[idx]
            
            # Visualize tokens and attention mask (after tokenization)
            f.write("**3. After Tokenization:**\n")
            vis = visualize_attention_mask(
                sample["input_ids"], 
                sample["attention_mask"], 
                tokenizer, 
                f"SmolTalk Example {idx+1} - Tokenized with Attention Mask"
            )
            f.write(vis)
            f.write("\n\n")
            
            # Check mask alignment
            alignment = check_mask_alignment(sample["input_ids"], sample["attention_mask"], tokenizer, formatted_text)
            f.write("**4. Attention Mask Analysis:**\n")
            f.write(f"- Assistant marker position in text: {alignment['text_boundary_pos']}\n")
            f.write(f"- First token with mask=1: {alignment['first_one_idx']}\n")
            f.write(f"- Has clear boundary: {'Yes' if alignment['has_clear_boundary'] else 'No'}\n")
            f.write(f"- Percentage of tokens with mask=1: {alignment['percent_ones']:.2f}%\n")
            
            f.write("\n" + "-"*70 + "\n\n")
        
        # 1.2 Debug UltraFeedback dataset (DPO training)
        f.write("### 1.2 UltraFeedback Dataset (DPO Training)\n\n")
        
        # Initialize dataset
        ultrafeedback_dataset = PreferenceDataset(
            dataset_name="ultrafeedback",
            tokenizer=tokenizer,
            max_length=512,
            subset_size=3  # Small sample for debugging
        )
        
        # Get examples from UltraFeedback
        for idx in range(min(3, len(ultrafeedback_dataset))):
            # Get original data item
            item = ultrafeedback_dataset.dataset[idx]
            
            # Get the raw data
            f.write(f"#### Example {idx+1}\n")
            f.write("**1. Original Data:**\n```json\n")
            f.write(f"prompt: {item['prompt']}\n")
            f.write(f"chosen: {item['chosen']}\n")
            f.write(f"rejected: {item['rejected']}\n")
            f.write("\n```\n\n")
            
            # Format conversations for chosen and rejected
            messages_chosen = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["chosen"]}
            ]
            
            messages_rejected = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["rejected"]}
            ]
            
            # Get formatted text (after _format_conversation)
            chosen_text = ultrafeedback_dataset._format_conversation(messages_chosen)
            rejected_text = ultrafeedback_dataset._format_conversation(messages_rejected)
            
            f.write("**2. After _format_conversation:**\n")
            f.write("Chosen Text:\n```\n")
            f.write(chosen_text)
            f.write("\n```\n\n")
            
            f.write("Rejected Text:\n```\n")
            f.write(rejected_text)
            f.write("\n```\n\n")
            
            # Get tokenized sample
            sample = ultrafeedback_dataset[idx]
            
            # Create a batch
            batch = {
                "chosen_input_ids": sample["chosen_input_ids"].unsqueeze(0),
                "rejected_input_ids": sample["rejected_input_ids"].unsqueeze(0),
                "chosen_attention_mask": sample["chosen_attention_mask"].unsqueeze(0),
                "rejected_attention_mask": sample["rejected_attention_mask"].unsqueeze(0)
            }
            
            # 3. Display tokenized data with attention masks
            f.write("**3. After Tokenization:**\n")
            
            # Visualize chosen tokens and attention mask
            vis_chosen = visualize_attention_mask(
                sample["chosen_input_ids"], 
                sample["chosen_attention_mask"], 
                tokenizer, 
                f"UltraFeedback Example {idx+1} - Chosen - Tokenized with Attention Mask"
            )
            f.write(vis_chosen)
            f.write("\n\n")
            
            # Visualize rejected tokens and attention mask
            vis_rejected = visualize_attention_mask(
                sample["rejected_input_ids"], 
                sample["rejected_attention_mask"], 
                tokenizer, 
                f"UltraFeedback Example {idx+1} - Rejected - Tokenized with Attention Mask"
            )
            f.write(vis_rejected)
            f.write("\n\n")
            
            # Apply DPO mask fixes
            fixed_batch = fix_dpo_masks(batch)
            
            # 4. Analyze original vs fixed attention masks
            f.write("**4. DPO Mask Analysis:**\n")
            
            # Check DPO mask fixes
            dpo_fixes = check_dpo_mask_fixes(batch)
            
            f.write("Original vs Fixed Masks:\n")
            f.write(f"- Split point (where chosen/rejected first differ): {dpo_fixes['split_point']}\n")
            f.write(f"- Original chosen first mask=1: {dpo_fixes['orig_chosen_first_one']}\n")
            f.write(f"- Fixed chosen first mask=1: {dpo_fixes['fixed_chosen_first_one']}\n")
            f.write(f"- Chosen attention mask changes: {dpo_fixes['chosen_mask_changes']}\n")
            f.write(f"- Chosen mask properly aligned with split point? {'Yes' if dpo_fixes['chosen_properly_aligned'] else 'No'}\n")
            
            f.write("\n" + "-"*70 + "\n\n")
        
        #############################################################
        # PART 2: EVALUATION PIPELINE DEBUGGING
        #############################################################
        f.write("## 2. EVALUATION PIPELINE DEBUGGING\n\n")
        
        # Load evaluation prompts
        eval_prompts = load_eval_prompts(num_prompts=3)  # Just use 3 prompts for debugging
        
        f.write("### 2.1 Evaluation Dataset\n\n")
        
        # Show the raw prompts
        for i, prompt in enumerate(eval_prompts[:3]):
            f.write(f"#### Evaluation Prompt {i+1}\n")
            f.write("**1. Original Prompt:**\n```\n")
            f.write(prompt)
            f.write("\n```\n\n")
            
            # Format the prompt as it would be in the evaluation pipeline
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = ultrafeedback_dataset._format_conversation(messages)
            
            f.write("**2. After _format_conversation:**\n```\n")
            f.write(formatted_prompt)
            f.write("\n```\n\n")
            
            # Tokenize the prompt
            encoded_prompt = tokenizer(formatted_prompt, return_tensors="pt")
            
            # Visualize tokens and attention mask
            f.write("**3. After Tokenization:**\n")
            vis_prompt = visualize_attention_mask(
                encoded_prompt["input_ids"][0], 
                encoded_prompt["attention_mask"][0], 
                tokenizer, 
                f"Evaluation Prompt {i+1} - Tokenized"
            )
            f.write(vis_prompt)
            f.write("\n\n" + "-"*70 + "\n\n")
        
        f.write("### 2.2 Generation and Reward Calculation\n\n")
        
        # Load a small model for demonstration
        try:
            # Try to load a small model for demonstration
            model_path = "Qwen/Qwen2.5-0.5B"
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Process just the first prompt
            prompt = eval_prompts[0]
            f.write(f"#### Processing Evaluation Prompt 1\n")
            f.write("**1. Original Prompt:**\n```\n")
            f.write(prompt)
            f.write("\n```\n\n")
            
            # Format the prompt
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = ultrafeedback_dataset._format_conversation(messages)
            
            f.write("**2. Formatted Prompt:**\n```\n")
            f.write(formatted_prompt)
            f.write("\n```\n\n")
            
            # Tokenize the prompt
            encoded_prompt = tokenizer(formatted_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                encoded_prompt = {k: v.to("cuda") for k, v in encoded_prompt.items()}
                model = model.to("cuda")
            
            # Generate a short response for demonstration
            with torch.no_grad():
                outputs = model.generate(
                    encoded_prompt["input_ids"],
                    max_new_tokens=20,  # Short for demo
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part
            response = generated_text[len(formatted_prompt):].strip()
            
            f.write("**3. Generated Response:**\n```\n")
            f.write(response)
            f.write("\n```\n\n")
            
            # Format the full response as it would appear after generation
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            f.write("**4. Full Generated Text (with special tokens):**\n```\n")
            f.write(full_response)
            f.write("\n```\n\n")
            
            # Explain the reward calculation process
            f.write("**5. Reward Calculation Process:**\n")
            f.write("In the actual evaluation, the following steps would be performed:\n\n")
            f.write("1. The model response would be extracted from the full generated text\n")
            f.write("2. The prompt and response would be formatted for the reward model API:\n")
            f.write("   ```json\n")
            f.write("   {\n")
            f.write('     "messages": [\n')
            f.write('       {"role": "user", "content": "' + prompt.replace('"', '\\"') + '"},\n')
            f.write('       {"role": "assistant", "content": "' + response.replace('"', '\\"') + '"}\n')
            f.write('     ]\n')
            f.write("   }\n")
            f.write("   ```\n\n")
            f.write("3. The Nemotron reward model would calculate a reward score\n")
            f.write("4. The same process would be repeated for the reference model\n")
            f.write("5. The two reward scores would be compared to determine a 'win'\n")
        except Exception as e:
            f.write(f"Error loading model for generation demo: {str(e)}\n")
            f.write("This part of the debugging requires a model to be loaded.\n")
        
        # Summary
        f.write("\n## 3. PIPELINE SUMMARY\n\n")
        f.write("### Training Pipeline:\n")
        f.write("1. Raw dataset → Format conversations → Tokenize → Set attention masks\n")
        f.write("2. For SFT: Forward pass with input_ids, attention_mask, and labels\n")
        f.write("3. For DPO: Process both chosen and rejected examples, compare reward scores\n\n")
        
        f.write("### Evaluation Pipeline:\n")
        f.write("1. Raw prompts → Format prompt → Tokenize → Generate response\n")
        f.write("2. Extract response → Format for reward model → Calculate reward\n")
        f.write("3. Compare against reference model → Calculate win rate and metrics\n\n")
        
        f.write("### Key Points:\n")
        f.write("- Attention masks are critical for proper loss calculation during training\n")
        f.write("- Proper formatting ensures consistent token boundaries between prompt and response\n")
        f.write("- The evaluation process mimics real-world usage while providing quantitative metrics\n")

if __name__ == "__main__":
    debug_attention_masks()
    print(f"Pipeline debugging complete. Results written to pipeline_debug_output.txt") 