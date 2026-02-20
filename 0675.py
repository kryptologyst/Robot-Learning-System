#!/usr/bin/env python3
"""Project 675: Robot Learning System - Modernized Demo

This is a demonstration of the modernized robot learning system.
The original simple Q-learning implementation has been replaced with a
comprehensive, production-ready framework.

For the full modern implementation, see the src/ directory and use the
provided scripts and demos.

This file demonstrates the basic usage of the new system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from algorithms.qlearning import QLearningAgent
from environments.gridworld import create_default_gridworld
from utils import set_seed, setup_logging, Timer
from visualization import LearningVisualizer


def main() -> None:
    """Demonstrate the modernized robot learning system."""
    print("🤖 Robot Learning System - Modernized Demo")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create environment
    print("Creating grid world environment...")
    env = create_default_gridworld()
    print(f"Environment: {env.grid_size[0]}x{env.grid_size[1]} grid with obstacles")
    
    # Create Q-learning agent
    print("Initializing Q-Learning agent...")
    agent = QLearningAgent(
        state_size=env.grid_size,
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42,
    )
    
    # Training
    print("Starting training...")
    with Timer("Training"):
        training_stats = agent.train(
            env=env,
            num_episodes=500,
            eval_frequency=100,
            verbose=True,
        )
    
    # Evaluation
    print("Evaluating trained agent...")
    with Timer("Evaluation"):
        eval_stats = agent.evaluate(env, num_episodes=50)
    
    # Results
    print("\n📊 Results:")
    print(f"  Success Rate: {eval_stats['success_rate']:.3f}")
    print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"  Mean Episode Length: {eval_stats['mean_length']:.1f}")
    
    # Visualization
    print("\n📈 Creating visualizations...")
    visualizer = LearningVisualizer()
    
    # Learning curves
    visualizer.plot_learning_curves(
        episode_rewards=training_stats["episode_rewards"],
        episode_lengths=training_stats["episode_lengths"],
        success_rates=training_stats["success_rates"],
        title="Q-Learning Training Progress",
    )
    
    # Policy visualization
    visualizer.plot_policy(
        q_table=agent.get_q_values(),
        grid_size=env.grid_size,
        obstacles=list(env.obstacles),
        start_position=tuple(env.start_position),
        goal_position=tuple(env.goal_position),
        title="Learned Policy",
    )
    
    print("\n✅ Demo completed successfully!")
    print("\nFor more advanced features, use:")
    print("  - python scripts/train_qlearning.py --help")
    print("  - python scripts/compare_algorithms.py --help")
    print("  - streamlit run demo/app.py")
    print("\n⚠️  DISCLAIMER: This is for research/education only.")
    print("   Do not use on real robots without expert review and safety measures.")


if __name__ == "__main__":
    main()
