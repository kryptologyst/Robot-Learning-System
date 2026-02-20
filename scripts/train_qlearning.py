#!/usr/bin/env python3
"""Training script for Q-Learning robot navigation.

This script demonstrates training a Q-Learning agent on a grid world environment
with comprehensive logging, evaluation, and visualization.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms.qlearning import QLearningAgent
from environments.gridworld import create_default_gridworld, create_simple_gridworld
from evaluation import EvaluationMetrics
from utils import set_seed, setup_logging, Timer, ensure_dir
from visualization import LearningVisualizer


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Q-Learning agent for robot navigation")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--discount-factor", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum exploration rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--env", choices=["simple", "default"], default="default", help="Environment type")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--save-model", action="store_true", help="Save trained model")
    parser.add_argument("--save-plots", action="store_true", help="Save visualization plots")
    parser.add_argument("--output-dir", type=str, default="data/logs", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    ensure_dir(args.output_dir)
    log_file = Path(args.output_dir) / "training.log"
    setup_logging(level=args.log_level, log_file=str(log_file))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Q-Learning training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create environment
    if args.env == "simple":
        env = create_simple_gridworld()
        logger.info("Using simple grid world environment")
    else:
        env = create_default_gridworld()
        logger.info("Using default grid world environment with obstacles")
    
    # Create agent
    agent = QLearningAgent(
        state_size=env.grid_size,
        action_size=env.action_space.n,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=args.seed,
    )
    
    # Create evaluation metrics tracker
    metrics = EvaluationMetrics()
    
    # Training loop
    with Timer("Training"):
        training_stats = agent.train(
            env=env,
            num_episodes=args.episodes,
            eval_frequency=100,
            verbose=True,
        )
        
        # Add training data to metrics
        for i, (reward, length) in enumerate(zip(training_stats["episode_rewards"], 
                                                training_stats["episode_lengths"])):
            success = reward > 0  # Simple success criterion
            metrics.add_episode(
                episode=i,
                reward=reward,
                length=length,
                success=success,
                algorithm="Q-Learning",
            )
    
    # Evaluation
    logger.info("Starting evaluation")
    with Timer("Evaluation"):
        eval_stats = agent.evaluate(env, num_episodes=args.eval_episodes)
    
    # Print results
    logger.info("Training Results:")
    logger.info(f"  Final Success Rate: {eval_stats['success_rate']:.3f}")
    logger.info(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    logger.info(f"  Mean Episode Length: {eval_stats['mean_length']:.1f} ± {eval_stats['std_length']:.1f}")
    
    # Create visualizations
    visualizer = LearningVisualizer()
    
    if args.save_plots:
        plots_dir = Path(args.output_dir) / "plots"
        ensure_dir(str(plots_dir))
        
        # Learning curves
        visualizer.plot_learning_curves(
            episode_rewards=training_stats["episode_rewards"],
            episode_lengths=training_stats["episode_lengths"],
            success_rates=training_stats["success_rates"],
            title="Q-Learning Training Progress",
            save_path=str(plots_dir / "learning_curves.png"),
        )
        
        # Policy visualization
        visualizer.plot_policy(
            q_table=agent.get_q_values(),
            grid_size=env.grid_size,
            obstacles=list(env.obstacles) if hasattr(env, 'obstacles') else None,
            start_position=tuple(env.start_position),
            goal_position=tuple(env.goal_position),
            title="Learned Policy",
            save_path=str(plots_dir / "policy.png"),
        )
        
        # Q-values heatmap
        visualizer.plot_q_values(
            q_table=agent.get_q_values(),
            grid_size=env.grid_size,
            title="Q-Values Heatmap",
            save_path=str(plots_dir / "q_values.png"),
        )
        
        logger.info(f"Saved plots to {plots_dir}")
    
    # Save model if requested
    if args.save_model:
        models_dir = Path(args.output_dir) / "models"
        ensure_dir(str(models_dir))
        model_path = models_dir / "qlearning_agent.npz"
        agent.save(str(model_path))
        logger.info(f"Saved model to {model_path}")
    
    # Save evaluation results
    results_path = Path(args.output_dir) / "evaluation_results.json"
    metrics.save_results(str(results_path))
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
