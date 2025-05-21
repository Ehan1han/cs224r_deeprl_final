import os
import argparse
import wandb
import json
from rl_llm.evaluation import run_evaluation
from rl_llm.model_utils import QwenModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--nemotron_api_key", type=str, default="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")
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--detailed_logging", action="store_true", help="Enable detailed per-prompt logging")
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if not args.output_dir:
        model_name = args.model_path.split('/')[-2]
        args.output_dir = f"outputs/eval/{model_name}"
    
    # Initialize wandb if enabled
    if args.use_wandb:
        run = wandb.init(
            # Set the entity
            entity="zhao111han-stanford-university",
            # Set the project
            project="cs224r_deeprl_final",
            # Name this run for better tracking
            name=f"eval_{args.model_path.split('/')[-2]}",
            # Track parameters
            config={
                "model_path": args.model_path,
                "num_prompts": args.num_prompts,
                "method": "eval",
                "detailed_logging": args.detailed_logging
            }
        )
    
    # Run evaluation
    print(f"\nEvaluating model: {args.model_path}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Using Nemotron API key: {args.nemotron_api_key[:10]}...")
    
    # Run evaluation with real-time wandb logging
    metrics = run_evaluation(
        model=args.model_path,  # Just pass the path, the evaluation code will handle it
        nemotron_api_key=args.nemotron_api_key,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb  # Enable real-time wandb logging
    )
    
    # Print only vital metrics
    print("\n----- EVALUATION RESULTS -----")
    print(f"Win rate:            {metrics['win_rate']:.4f}")
    print(f"Avg model reward:    {metrics['avg_model_reward']:.4f}")
    print(f"Avg reference reward: {metrics['avg_ref_reward']:.4f}")
    print(f"Reward improvement:   {metrics['reward_improvement']:.4f}")
    print("------------------------------")
    
    # Log to wandb if enabled
    if args.use_wandb:
        # Create enhanced wandb metrics
        wandb_metrics = {
            "eval/win_rate": metrics["win_rate"],
            "eval/avg_model_reward": metrics["avg_model_reward"],
            "eval/avg_ref_reward": metrics["avg_ref_reward"],
            "eval/reward_improvement": metrics["reward_improvement"]
        }
        
        # Add histogram data if available
        if "model_rewards" in metrics and "ref_rewards" in metrics:
            wandb_metrics["eval/model_rewards_histogram"] = wandb.Histogram(metrics["model_rewards"])
            wandb_metrics["eval/ref_rewards_histogram"] = wandb.Histogram(metrics["ref_rewards"])
        
        # Log prompt-wise detailed results if available and detailed logging is enabled
        if "prompt_results" in metrics and args.detailed_logging:
            # Create a table for the prompts and responses
            columns = ["prompt", "model_response", "ref_response", "model_reward", "ref_reward", "win"]
            prompt_table = wandb.Table(columns=columns)
            
            # Add rows to the table
            for result in metrics["prompt_results"]:
                prompt = result.get("prompt", "")
                model_response = result.get("model_response", "")
                ref_response = result.get("ref_response", "")
                model_reward = result.get("model_reward", 0)
                ref_reward = result.get("ref_reward", 0)
                win = result.get("win", False)
                
                prompt_table.add_data(
                    prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    model_response[:100] + "..." if len(model_response) > 100 else model_response,
                    ref_response[:100] + "..." if len(ref_response) > 100 else ref_response,
                    model_reward,
                    ref_reward,
                    "Win" if win else "Loss"
                )
            
            # Log the table
            wandb_metrics["eval/prompt_results"] = prompt_table
        
        # Log all metrics
        wandb.log(wandb_metrics)
        run.finish()

if __name__ == "__main__":
    main() 