#!/usr/bin/env python3
"""Algorithm comparison script for robot learning system.

This script compares different RL algorithms on the same environment
and generates comprehensive evaluation reports.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms.qlearning import QLearningAgent
from environments.gridworld import create_default_gridworld, create_simple_gridworld
from evaluation import EvaluationMetrics, compare_algorithms
from utils import set_seed, setup_logging, Timer, ensure_dir
from visualization import LearningVisualizer, create_leaderboard_plot


def run_algorithm_comparison(
    algorithms: List[str],
    env_type: str,
    num_runs: int,
    num_episodes: int,
    eval_episodes: int,
    seed: int,
    output_dir: str,
) -> None:
    """Run comprehensive algorithm comparison.

    Args:
        algorithms: List of algorithm names to compare
        env_type: Environment type ('simple' or 'default')
        num_runs: Number of independent runs per algorithm
        num_episodes: Number of training episodes
        eval_episodes: Number of evaluation episodes
        seed: Random seed
        output_dir: Output directory for results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting algorithm comparison with {len(algorithms)} algorithms")
    logger.info(f"Environment: {env_type}, Runs: {num_runs}, Episodes: {num_episodes}")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Results storage
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    metrics_tracker = EvaluationMetrics()
    
    # Create environment
    if env_type == "simple":
        env = create_simple_gridworld()
    else:
        env = create_default_gridworld()
    
    for algo_name in algorithms:
        logger.info(f"Training {algo_name}")
        algo_results = []
        
        for run in range(num_runs):
            logger.info(f"  Run {run + 1}/{num_runs}")
            
            # Set unique seed for each run
            run_seed = seed + run
            
            # Create agent
            if algo_name == "Q-Learning":
                agent = QLearningAgent(
                    state_size=env.grid_size,
                    action_size=env.action_space.n,
                    learning_rate=0.1,
                    discount_factor=0.9,
                    epsilon=0.1,
                    epsilon_decay=0.995,
                    epsilon_min=0.01,
                    seed=run_seed,
                )
            else:
                logger.warning(f"Unknown algorithm: {algo_name}, skipping")
                continue
            
            # Training
            with Timer(f"{algo_name} Run {run + 1} Training"):
                training_stats = agent.train(
                    env=env,
                    num_episodes=num_episodes,
                    eval_frequency=100,
                    verbose=False,
                )
            
            # Evaluation
            with Timer(f"{algo_name} Run {run + 1} Evaluation"):
                eval_stats = agent.evaluate(env, num_episodes=eval_episodes)
            
            # Store results
            run_results = {
                "algorithm": algo_name,
                "run": run,
                "seed": run_seed,
                "training_rewards": training_stats["episode_rewards"],
                "training_lengths": training_stats["episode_lengths"],
                "eval_mean_reward": eval_stats["mean_reward"],
                "eval_std_reward": eval_stats["std_reward"],
                "eval_success_rate": eval_stats["success_rate"],
                "eval_mean_length": eval_stats["mean_length"],
                "final_epsilon": agent.epsilon,
            }
            
            algo_results.append(run_results)
            
            # Add to metrics tracker
            for i, (reward, length) in enumerate(zip(training_stats["episode_rewards"], 
                                                   training_stats["episode_lengths"])):
                success = reward > 0
                metrics_tracker.add_episode(
                    episode=i,
                    reward=reward,
                    length=length,
                    success=success,
                    algorithm=algo_name,
                    run=run,
                )
        
        all_results[algo_name] = algo_results
    
    # Generate comparison report
    logger.info("Generating comparison report")
    
    # Create comparison DataFrame
    comparison_data = []
    for algo_name, runs in all_results.items():
        eval_rewards = [run["eval_mean_reward"] for run in runs]
        success_rates = [run["eval_success_rate"] for run in runs]
        
        comparison_data.append({
            "Algorithm": algo_name,
            "Mean Reward": np.mean(eval_rewards),
            "Std Reward": np.std(eval_rewards),
            "Min Reward": np.min(eval_rewards),
            "Max Reward": np.max(eval_rewards),
            "Mean Success Rate": np.mean(success_rates),
            "Std Success Rate": np.std(success_rates),
            "Runs": len(runs),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("Mean Reward", ascending=False)
    
    # Save results
    results_path = Path(output_dir) / "comparison_results.csv"
    comparison_df.to_csv(results_path, index=False)
    logger.info(f"Saved comparison results to {results_path}")
    
    # Generate leaderboard
    leaderboard = metrics_tracker.get_leaderboard(algorithms)
    leaderboard_path = Path(output_dir) / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logger.info(f"Saved leaderboard to {leaderboard_path}")
    
    # Create visualizations
    visualizer = LearningVisualizer()
    
    # Learning curves comparison
    fig_data = {}
    for algo_name, runs in all_results.items():
        all_rewards = []
        for run in runs:
            all_rewards.extend(run["training_rewards"])
        fig_data[algo_name] = all_rewards
    
    visualizer.plot_algorithm_comparison(
        fig_data,
        metric="reward",
        title="Algorithm Comparison - Training Rewards",
        save_path=str(Path(output_dir) / "algorithm_comparison.png"),
    )
    
    # Create leaderboard plot
    leaderboard_data = {}
    for algo_name, runs in all_results.items():
        eval_rewards = [run["eval_mean_reward"] for run in runs]
        leaderboard_data[algo_name] = {
            "success_rate": np.mean([run["eval_success_rate"] for run in runs]),
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
        }
    
    create_leaderboard_plot(
        leaderboard_data,
        metric="success_rate",
        title="Algorithm Leaderboard - Success Rate",
        save_path=str(Path(output_dir) / "leaderboard_plot.png"),
    )
    
    # Print summary
    logger.info("Comparison Summary:")
    logger.info("=" * 50)
    for _, row in comparison_df.iterrows():
        logger.info(f"{row['Algorithm']}:")
        logger.info(f"  Mean Reward: {row['Mean Reward']:.3f} ± {row['Std Reward']:.3f}")
        logger.info(f"  Success Rate: {row['Mean Success Rate']:.3f} ± {row['Std Success Rate']:.3f}")
        logger.info(f"  Runs: {row['Runs']}")
        logger.info("")
    
    # Save detailed results
    detailed_results = {
        "comparison_summary": comparison_df.to_dict("records"),
        "algorithm_results": all_results,
        "environment": {
            "type": env_type,
            "grid_size": env.grid_size,
            "obstacles": list(env.obstacles) if hasattr(env, 'obstacles') else [],
        },
        "experiment_config": {
            "num_runs": num_runs,
            "num_episodes": num_episodes,
            "eval_episodes": eval_episodes,
            "seed": seed,
        },
    }
    
    import json
    detailed_path = Path(output_dir) / "detailed_results.json"
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    logger.info(f"Saved detailed results to {detailed_path}")
    logger.info("Algorithm comparison completed successfully!")


def main() -> None:
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare RL algorithms")
    parser.add_argument("--algorithms", nargs="+", default=["Q-Learning"], 
                       help="Algorithms to compare")
    parser.add_argument("--env", choices=["simple", "default"], default="default",
                       help="Environment type")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/comparison", 
                       help="Output directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    ensure_dir(args.output_dir)
    log_file = Path(args.output_dir) / "comparison.log"
    setup_logging(level=args.log_level, log_file=str(log_file))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting algorithm comparison")
    logger.info(f"Arguments: {vars(args)}")
    
    # Run comparison
    run_algorithm_comparison(
        algorithms=args.algorithms,
        env_type=args.env,
        num_runs=args.runs,
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
