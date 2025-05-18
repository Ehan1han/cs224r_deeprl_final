from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from vllm import LLM, SamplingParams
from openai import OpenAI
import os
from tqdm import tqdm
from .model_utils import QwenModel

class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        reference_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        nemotron_api_key: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model
            reference_model_path: Path to the reference model
            nemotron_api_key: API key for Nemotron reward model
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
        """
        self.device = device
        self.batch_size = batch_size
        
        # Initialize VLLM models
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9
        )
        self.reference_model = LLM(
            model=reference_model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9
        )
        
        # Initialize Nemotron client
        self.nemotron_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nemotron_api_key or os.getenv("NEMOTRON_API_KEY")
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=None
        )
    
    def _get_reward_score(self, prompt: str, response: str) -> float:
        """
        Get reward score from Nemotron model.
        
        Args:
            prompt: Input prompt
            response: Model response
            
        Returns:
            Reward score
        """
        completion = self.nemotron_client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-reward",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        )
        return float(completion.choices[0].message.content)
    
    def _generate_responses(
        self,
        prompts: List[str],
        model: LLM
    ) -> List[str]:
        """
        Generate responses using VLLM.
        
        Args:
            prompts: List of prompts
            model: VLLM model
            
        Returns:
            List of generated responses
        """
        outputs = model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def evaluate(
        self,
        eval_prompts: List[str],
        num_samples: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate model against reference model.
        
        Args:
            eval_prompts: List of prompts to evaluate
            num_samples: Number of samples per prompt
            
        Returns:
            Dictionary containing evaluation metrics
        """
        wins = 0
        ties = 0
        total_rewards = []
        reference_rewards = []
        model_lengths = []
        ref_lengths = []
        
        # Process prompts in batches
        for i in tqdm(range(0, len(eval_prompts), self.batch_size)):
            batch_prompts = eval_prompts[i:i + self.batch_size]
            
            # Generate responses
            model_responses = self._generate_responses(batch_prompts, self.model)
            reference_responses = self._generate_responses(batch_prompts, self.reference_model)
            
            # Get reward scores
            for prompt, model_response, ref_response in zip(
                batch_prompts, model_responses, reference_responses
            ):
                model_reward = self._get_reward_score(prompt, model_response)
                ref_reward = self._get_reward_score(prompt, ref_response)
                
                total_rewards.append(model_reward)
                reference_rewards.append(ref_reward)
                model_lengths.append(len(model_response.split()))
                ref_lengths.append(len(ref_response.split()))
                
                if model_reward > ref_reward:
                    wins += 1
                elif abs(model_reward - ref_reward) < 0.1:  # Consider rewards within 0.1 as ties
                    ties += 1
        
        # Calculate metrics
        total_comparisons = len(eval_prompts)
        win_rate = wins / total_comparisons
        tie_rate = ties / total_comparisons
        avg_reward = np.mean(total_rewards)
        avg_reference_reward = np.mean(reference_rewards)
        avg_model_length = np.mean(model_lengths)
        avg_ref_length = np.mean(ref_lengths)
        
        return {
            "win_rate": win_rate,
            "tie_rate": tie_rate,
            "avg_reward": avg_reward,
            "avg_reference_reward": avg_reference_reward,
            "reward_improvement": avg_reward - avg_reference_reward,
            "avg_model_length": avg_model_length,
            "avg_ref_length": avg_ref_length,
            "length_ratio": avg_model_length / avg_ref_length
        }

def load_eval_prompts(
    dataset_name: str = "ultrafeedback",
    split: str = "test",
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
    nemotron_api_key: Optional[str] = None,
    num_prompts: int = 100,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Run full evaluation pipeline.
    
    Args:
        model_path: Path to the trained model
        nemotron_api_key: API key for Nemotron reward model
        num_prompts: Number of prompts to evaluate
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load evaluation prompts
    eval_prompts = load_eval_prompts(num_prompts=num_prompts)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        nemotron_api_key=nemotron_api_key
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(eval_prompts)
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        import json
        with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics 