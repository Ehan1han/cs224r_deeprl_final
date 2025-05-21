from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import json
from tqdm import tqdm
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        reference_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        nemotron_api_key: str = "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",
        use_wandb: bool = False
    ):
        """
        Initialize the model evaluator using OpenAI API for Nemotron reward scoring.
        
        Args:
            model_path: Path to the trained model
            reference_model_path: Path to the reference model
            nemotron_api_key: API key for Nemotron reward model
            use_wandb: Whether to log to wandb during evaluation
        """
        self.model_path = model_path
        self.reference_model_path = reference_model_path
        self.api_key = nemotron_api_key
        self.use_wandb = use_wandb
        
        # Initialize Nemotron client using the exact format from the documentation
        self.nemotron_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nemotron_api_key
        )
        
        # Initialize both models only once
        print(f"Loading model: {model_path}")
        self.model, self.model_tokenizer = self._load_model(model_path)
        
        print(f"Loading reference model: {reference_model_path}")
        self.ref_model, self.ref_tokenizer = self._load_model(reference_model_path)
    
    def _load_model(self, model_path):
        """Helper function to load model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if torch.cuda.is_available():
            model = model.to("cuda")
        return model, tokenizer
    
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
    
    def generate_response(self, prompt: str, use_model=True) -> str:
        """
        Generate a response from a model for the given prompt.
        Uses pre-loaded models instead of loading for each prompt.
        
        Args:
            prompt: The input prompt
            use_model: If True, use self.model, otherwise use self.ref_model
            
        Returns:
            Generated response text
        """
        try:
            # Select the appropriate model and tokenizer
            model = self.model if use_model else self.ref_model
            tokenizer = self.model_tokenizer if use_model else self.ref_tokenizer
            
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
        prompt_results = []
        
        print(f"Evaluating {len(eval_prompts)} prompts...")
        for i, prompt in enumerate(tqdm(eval_prompts)):
            # Generate responses from both models
            model_response = self.generate_response(prompt, use_model=True)
            ref_response = self.generate_response(prompt, use_model=False)
            
            # Get reward scores using OpenAI API (Nemotron)
            model_reward = self.get_reward_score(prompt, model_response)
            ref_reward = self.get_reward_score(prompt, ref_response)
            
            # Track metrics
            model_rewards.append(model_reward)
            ref_rewards.append(ref_reward)
            
            # Count win
            is_win = model_reward > ref_reward
            if is_win:
                wins += 1
            
            # Store result for this prompt
            prompt_result = {
                "prompt": prompt,
                "model_response": model_response,
                "ref_response": ref_response,
                "model_reward": model_reward,
                "ref_reward": ref_reward,
                "win": is_win
            }
            prompt_results.append(prompt_result)
            
            # Log real-time metrics to wandb
            if self.use_wandb:
                try:
                    import wandb
                    current_win_rate = wins / (i + 1)
                    current_avg_model_reward = np.mean(model_rewards)
                    current_avg_ref_reward = np.mean(ref_rewards)
                    current_reward_improvement = current_avg_model_reward - current_avg_ref_reward
                    
                    wandb.log({
                        "current_win_rate": current_win_rate,
                        "current_avg_model_reward": current_avg_model_reward,
                        "current_avg_ref_reward": current_avg_ref_reward,
                        "current_reward_improvement": current_reward_improvement,
                        "examples_evaluated": i + 1,
                        "last_model_reward": model_reward,
                        "last_ref_reward": ref_reward,
                        "last_win": is_win
                    })
                except Exception as e:
                    print(f"Error logging to wandb: {str(e)}")
            
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
            "reward_improvement": reward_improvement,
            "model_rewards": model_rewards,
            "ref_rewards": ref_rewards,
            "prompt_results": prompt_results
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
    model,  # Can be either a model instance or path string
    nemotron_api_key: str = "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",
    num_prompts: int = 100,
    output_dir: Optional[str] = None,
    use_wandb: bool = False,
    model_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Run full evaluation pipeline.
    
    Args:
        model: Either a model instance or path to the trained model
        nemotron_api_key: API key for Nemotron reward model (default provided)
        num_prompts: Number of prompts to evaluate
        output_dir: Directory to save evaluation results
        use_wandb: Whether to log to wandb during evaluation
        model_path: Path to model (required if model is an instance)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load evaluation prompts
    print(f"Loading {num_prompts} evaluation prompts...")
    eval_prompts = load_eval_prompts(split="test_prefs", num_prompts=num_prompts)
    
    # If model is a string (path), use it directly
    if isinstance(model, str):
        model_path = model
        model_instance = None
    else:
        # Otherwise use the provided model instance
        model_instance = model
        if model_path is None:
            raise ValueError("model_path must be provided when passing a model instance")
    
    # Initialize evaluator with OpenAI API
    evaluator = ModelEvaluator(
        model_path=model_path,
        nemotron_api_key=nemotron_api_key,
        use_wandb=use_wandb
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
        
        # Save basic metrics
        basic_metrics = {
            "win_rate": metrics["win_rate"],
            "avg_model_reward": metrics["avg_model_reward"],
            "avg_ref_reward": metrics["avg_ref_reward"],
            "reward_improvement": metrics["reward_improvement"]
        }
        with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(basic_metrics, f, indent=2)
        
        # Save detailed results (optional)
        if "prompt_results" in metrics and len(metrics["prompt_results"]) > 0:
            with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
                # Save only the first 10 detailed results to avoid huge files
                json.dump(metrics["prompt_results"][:10], f, indent=2)
    
    return metrics 