import os
import argparse
import wandb
import json
from rl_llm.evaluation import run_evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--nemotron_api_key", type=str)
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
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
                "method": "eval"
            }
        )
    
    # Run evaluation
    metrics = run_evaluation(
        model_path=args.model_path,
        nemotron_api_key=args.nemotron_api_key,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir
    )
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log(metrics)
        run.finish()

if __name__ == "__main__":
    main() 