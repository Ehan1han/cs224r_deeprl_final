import os
import argparse
from rl_llm.evaluation import run_evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--nemotron_api_key", type=str)
    parser.add_argument("--num_prompts", type=int, default=100)
    args = parser.parse_args()
    
    # Run evaluation
    metrics = run_evaluation(
        model_path=args.model_path,
        nemotron_api_key=args.nemotron_api_key,
        num_prompts=args.num_prompts
    )
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Win Rate: {metrics['win_rate']:.4f}")
    print(f"Average Reward: {metrics['avg_reward']:.4f}")
    print(f"Average Reference Reward: {metrics['avg_reference_reward']:.4f}")
    print(f"Reward Improvement: {metrics['reward_improvement']:.4f}")

if __name__ == "__main__":
    main() 