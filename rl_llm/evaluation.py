from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import json
from tqdm import tqdm
from openai import OpenAI

class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        reference_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        nemotron_api_key: str = "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"
    ):
        """
        Initialize the model evaluator using OpenAI API for Nemotron reward scoring.
        
        Args:
            model_path: Path to the trained model
            reference_model_path: Path to the reference model
            nemotron_api_key: API key for Nemotron reward model
        """
        self.model_path = model_path
        self.reference_model_path = reference_model_path
        self.api_key = nemotron_api_key
        
        # Initialize Nemotron client using the exact format from the documentation
        self.nemotron_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nemotron_api_key
        )
    
    def get_reward_score(self, prompt: str, response: str) -> float:
        """
        Get reward score from Nemotron model using OpenAI API.
        
        Args:
            prompt: Input prompt
            response: Model response
            
        Returns:
            Reward score (higher is better)
        """
        try:
            # Use the exact format from the documentation
            completion = self.nemotron_client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-reward",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            )
            # Parse the reward from the format "reward:-25.625"
            reward_text = completion.choices[0].message.content
            print(f"Raw reward response: {reward_text}")
            
            # Extract the number after "reward:"
            if "reward:" in reward_text:
                reward_value = reward_text.split("reward:")[1].strip()
                return float(reward_value)
            else:
                return float(reward_text)  # Try direct conversion as fallback
            
        except Exception as e:
            print(f"Error getting reward score: {str(e)}")
            return 0.0
    
    def generate_response(self, prompt: str, model) -> str:
        """
        Generate a response from a model for the given prompt.
        This uses a non-API method since we want to evaluate our local trained models.
        
        Args:
            prompt: The input prompt
            model: The model to use for generation
            
        Returns:
            Generated response text
        """
        # Import here to avoid dependency issues if not using this method
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        try:
            # Load model and tokenizer
            if isinstance(model, str):
                print(f"Loading model: {model}")
                tokenizer = AutoTokenizer.from_pretrained(model)
                model = AutoModelForCausalLM.from_pretrained(model)
                if torch.cuda.is_available():
                    model = model.to("cuda")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            response = full_output[len(prompt):].strip()
            
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return ""
    
    def evaluate(self, eval_prompts: List[str]) -> Dict[str, float]:
        """
        Evaluate trained model against reference model using Nemotron reward scores.
        
        Args:
            eval_prompts: List of prompts to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        wins = 0
        model_rewards = []
        ref_rewards = []
        
        print(f"Evaluating {len(eval_prompts)} prompts...")
        for i, prompt in enumerate(tqdm(eval_prompts)):
            # Generate responses from both models
            model_response = self.generate_response(prompt, self.model_path)
            ref_response = self.generate_response(prompt, self.reference_model_path)
            
            # Get reward scores using OpenAI API (Nemotron)
            model_reward = self.get_reward_score(prompt, model_response)
            ref_reward = self.get_reward_score(prompt, ref_response)
            
            # Track metrics
            model_rewards.append(model_reward)
            ref_rewards.append(ref_reward)
            
            # Count win
            if model_reward > ref_reward:
                wins += 1
            
            # Progress logging
            if (i + 1) % 5 == 0 or i == 0:
                win_rate = wins / (i + 1)
                print(f"Progress: {i+1}/{len(eval_prompts)}, Current win rate: {win_rate:.4f}")
                print(f"Prompt: {prompt[:50]}...")
                print(f"Model output: {model_response[:50]}... (reward: {model_reward:.2f})")
                print(f"Reference output: {ref_response[:50]}... (reward: {ref_reward:.2f})")
        
        # Calculate final metrics
        win_rate = wins / len(eval_prompts)
        avg_model_reward = np.mean(model_rewards)
        avg_ref_reward = np.mean(ref_rewards)
        reward_improvement = avg_model_reward - avg_ref_reward
        
        return {
            "win_rate": win_rate,
            "avg_model_reward": avg_model_reward,
            "avg_ref_reward": avg_ref_reward,
            "reward_improvement": reward_improvement
        }

def load_eval_prompts(
    dataset_name: str = "ultrafeedback",
    split: str = "test_prefs",
    num_prompts: int = 100
) -> List[str]:
    """
    Load evaluation prompts from dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split to use
        num_prompts: Number of prompts to load
        
    Returns:
        List of prompts
    """
    from datasets import load_dataset
    
    if dataset_name == "ultrafeedback":
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
        prompts = dataset["prompt"][:num_prompts]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return prompts

def run_evaluation(
    model_path: str,
    nemotron_api_key: str = "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",
    num_prompts: int = 100,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Run full evaluation pipeline.
    
    Args:
        model_path: Path to the trained model
        nemotron_api_key: API key for Nemotron reward model (default provided)
        num_prompts: Number of prompts to evaluate
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load evaluation prompts
    print(f"Loading {num_prompts} evaluation prompts...")
    eval_prompts = load_eval_prompts(split="test_prefs", num_prompts=num_prompts)
    
    # Initialize evaluator with OpenAI API
    evaluator = ModelEvaluator(
        model_path=model_path,
        nemotron_api_key=nemotron_api_key
    )
    
    # Run evaluation
    print(f"Starting evaluation with Nemotron API...")
    metrics = evaluator.evaluate(eval_prompts)
    
    # Print results
    print("\n----- EVALUATION RESULTS -----")
    print(f"Win rate:            {metrics['win_rate']:.4f}")
    print(f"Avg model reward:    {metrics['avg_model_reward']:.4f}")
    print(f"Avg reference reward: {metrics['avg_ref_reward']:.4f}")
    print(f"Reward improvement:   {metrics['reward_improvement']:.4f}")
    print("------------------------------")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics 