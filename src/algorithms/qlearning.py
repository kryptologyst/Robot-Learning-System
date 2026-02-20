"""Q-Learning algorithm implementation for robot learning system.

This module implements a modern, type-safe Q-learning algorithm with proper
logging, evaluation metrics, and reproducibility features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from environments.gridworld import GridWorld
from utils.seeding import set_seed

logger = logging.getLogger(__name__)


class QLearningAgent:
    """Q-Learning agent for discrete state-action spaces.

    Implements tabular Q-learning with epsilon-greedy exploration and
    comprehensive logging and evaluation capabilities.

    Args:
        state_size: Size of the state space
        action_size: Number of possible actions
        learning_rate: Learning rate for Q-value updates
        discount_factor: Discount factor for future rewards
        epsilon: Initial exploration rate
        epsilon_decay: Rate of epsilon decay
        epsilon_min: Minimum exploration rate
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        state_size: Tuple[int, int],
        action_size: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table
        self.q_table = np.zeros((state_size[0], state_size[1], action_size), dtype=np.float32)

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.epsilon_history: List[float] = []
        self.q_value_history: List[float] = []

        # Set random seed
        if seed is not None:
            set_seed(seed)

        logger.info(f"Initialized Q-Learning agent with state size {state_size}, action size {action_size}")

    def get_action(self, state: NDArray[np.int32], training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state (position)
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_size)
        else:
            # Exploitation: best action according to Q-table
            x, y = state
            return int(np.argmax(self.q_table[x, y]))

    def update(
        self,
        state: NDArray[np.int32],
        action: int,
        reward: float,
        next_state: NDArray[np.int32],
        done: bool,
    ) -> None:
        """Update Q-table using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode is finished
        """
        x, y = state
        next_x, next_y = next_state

        # Current Q-value
        current_q = self.q_table[x, y, action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_x, next_y])

        # Q-learning update
        self.q_table[x, y, action] = current_q + self.learning_rate * (target_q - current_q)

        # Track Q-value changes
        self.q_value_history.append(abs(target_q - current_q))

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.epsilon_history.append(self.epsilon)

    def train_episode(self, env: GridWorld) -> Dict[str, float]:
        """Train the agent for one episode.

        Args:
            env: Environment to train on

        Returns:
            Episode statistics
        """
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = env.max_steps

        while steps < max_steps:
            # Select action
            action = self.get_action(state, training=True)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update Q-table
            self.update(state, action, reward, next_state, done)

            # Update state and statistics
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Decay exploration rate
        self.decay_epsilon()

        # Record episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)

        episode_stats = {
            "reward": total_reward,
            "length": steps,
            "epsilon": self.epsilon,
            "success": terminated,
        }

        return episode_stats

    def train(
        self,
        env: GridWorld,
        num_episodes: int = 1000,
        eval_frequency: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the agent for multiple episodes.

        Args:
            env: Environment to train on
            num_episodes: Number of training episodes
            eval_frequency: Frequency of evaluation logging
            verbose: Whether to print progress

        Returns:
            Training statistics
        """
        logger.info(f"Starting Q-Learning training for {num_episodes} episodes")

        training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "success_rates": [],
            "avg_rewards": [],
        }

        for episode in range(num_episodes):
            # Train one episode
            episode_stats = self.train_episode(env)

            # Record statistics
            training_stats["episode_rewards"].append(episode_stats["reward"])
            training_stats["episode_lengths"].append(episode_stats["length"])

            # Calculate running averages
            if episode >= 99:  # Start calculating averages after 100 episodes
                recent_rewards = training_stats["episode_rewards"][-100:]
                recent_successes = sum(1 for r in recent_rewards if r > 0)
                success_rate = recent_successes / min(100, episode + 1)
                avg_reward = np.mean(recent_rewards)

                training_stats["success_rates"].append(success_rate)
                training_stats["avg_rewards"].append(avg_reward)
            else:
                training_stats["success_rates"].append(0.0)
                training_stats["avg_rewards"].append(episode_stats["reward"])

            # Log progress
            if verbose and (episode + 1) % eval_frequency == 0:
                recent_rewards = training_stats["episode_rewards"][-eval_frequency:]
                avg_reward = np.mean(recent_rewards)
                success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)

                logger.info(
                    f"Episode {episode + 1}/{num_episodes} - "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Success Rate: {success_rate:.2f}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        logger.info("Q-Learning training completed")
        return training_stats

    def evaluate(self, env: GridWorld, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate the trained agent.

        Args:
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation statistics
        """
        logger.info(f"Evaluating Q-Learning agent for {num_episodes} episodes")

        eval_rewards = []
        eval_lengths = []
        successes = 0

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            steps = 0
            max_steps = env.max_steps

            while steps < max_steps:
                # Use greedy policy (no exploration)
                action = self.get_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    if terminated:
                        successes += 1
                    break

            eval_rewards.append(total_reward)
            eval_lengths.append(steps)

        eval_stats = {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "std_length": np.std(eval_lengths),
            "success_rate": successes / num_episodes,
            "min_reward": np.min(eval_rewards),
            "max_reward": np.max(eval_rewards),
        }

        logger.info(
            f"Evaluation Results - "
            f"Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}, "
            f"Success Rate: {eval_stats['success_rate']:.2f}, "
            f"Mean Length: {eval_stats['mean_length']:.1f}"
        )

        return eval_stats

    def get_policy(self) -> NDArray[np.int32]:
        """Get the learned policy (greedy actions).

        Returns:
            Policy matrix with best action for each state
        """
        return np.argmax(self.q_table, axis=2)

    def get_q_values(self) -> NDArray[np.float32]:
        """Get the Q-table.

        Returns:
            Q-value table
        """
        return self.q_table.copy()

    def save(self, filepath: str) -> None:
        """Save the trained agent.

        Args:
            filepath: Path to save the agent
        """
        agent_data = {
            "q_table": self.q_table,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "epsilon_history": self.epsilon_history,
        }

        np.savez(filepath, **agent_data)
        logger.info(f"Saved Q-Learning agent to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> QLearningAgent:
        """Load a trained agent.

        Args:
            filepath: Path to load the agent from

        Returns:
            Loaded Q-Learning agent
        """
        data = np.load(filepath)

        agent = cls(
            state_size=tuple(data["state_size"]),
            action_size=int(data["action_size"]),
            learning_rate=float(data["learning_rate"]),
            discount_factor=float(data["discount_factor"]),
            epsilon=float(data["epsilon"]),
            epsilon_decay=float(data["epsilon_decay"]),
            epsilon_min=float(data["epsilon_min"]),
        )

        agent.q_table = data["q_table"]
        agent.episode_rewards = data["episode_rewards"].tolist()
        agent.episode_lengths = data["episode_lengths"].tolist()
        agent.epsilon_history = data["epsilon_history"].tolist()

        logger.info(f"Loaded Q-Learning agent from {filepath}")
        return agent
