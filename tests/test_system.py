"""Test suite for robot learning system.

This module provides comprehensive unit tests for all components
of the robot learning system.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms.qlearning import QLearningAgent
from environments.gridworld import GridWorld, create_default_gridworld, create_simple_gridworld
from evaluation import EvaluationMetrics
from utils import set_seed, get_device, Timer, moving_average, calculate_confidence_interval


class TestGridWorld(unittest.TestCase):
    """Test cases for GridWorld environment."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.env = GridWorld(
            grid_size=(3, 3),
            start_position=(0, 0),
            goal_position=(2, 2),
            obstacles=[(1, 1)],
            max_steps=10,
            seed=42,
        )

    def test_initialization(self) -> None:
        """Test environment initialization."""
        self.assertEqual(self.env.grid_size, (3, 3))
        self.assertTrue(np.array_equal(self.env.start_position, np.array([0, 0])))
        self.assertTrue(np.array_equal(self.env.goal_position, np.array([2, 2])))
        self.assertEqual(self.env.obstacles, {(1, 1)})
        self.assertEqual(self.env.max_steps, 10)

    def test_reset(self) -> None:
        """Test environment reset."""
        obs, info = self.env.reset()
        self.assertTrue(np.array_equal(obs, np.array([0, 0])))
        self.assertEqual(info["step_count"], 0)
        self.assertEqual(info["episode_reward"], 0.0)

    def test_step(self) -> None:
        """Test environment step function."""
        obs, info = self.env.reset()
        
        # Test valid move
        obs, reward, terminated, truncated, info = self.env.step(3)  # Right
        self.assertTrue(np.array_equal(obs, np.array([0, 1])))
        self.assertEqual(reward, -0.1)  # Step penalty
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        
        # Test obstacle collision
        obs, reward, terminated, truncated, info = self.env.step(1)  # Down
        obs, reward, terminated, truncated, info = self.env.step(3)  # Right
        self.assertEqual(reward, -1.0)  # Obstacle penalty
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_goal_reaching(self) -> None:
        """Test reaching the goal."""
        obs, info = self.env.reset()
        
        # Move to goal
        obs, reward, terminated, truncated, info = self.env.step(3)  # Right
        obs, reward, terminated, truncated, info = self.env.step(3)  # Right
        obs, reward, terminated, truncated, info = self.env.step(1)  # Down
        obs, reward, terminated, truncated, info = self.env.step(1)  # Down
        
        self.assertEqual(reward, 10.0)  # Goal reward
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_out_of_bounds(self) -> None:
        """Test out of bounds movement."""
        obs, info = self.env.reset()
        
        # Try to move up from top row
        obs, reward, terminated, truncated, info = self.env.step(0)  # Up
        self.assertEqual(reward, -1.0)  # Out of bounds penalty
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_get_possible_actions(self) -> None:
        """Test getting possible actions."""
        obs, info = self.env.reset()
        
        # From corner, should have 2 possible actions
        possible_actions = self.env.get_possible_actions()
        self.assertIn(1, possible_actions)  # Down
        self.assertIn(3, possible_actions)   # Right
        self.assertNotIn(0, possible_actions)  # Up (out of bounds)
        self.assertNotIn(2, possible_actions)  # Left (out of bounds)

    def test_state_representation(self) -> None:
        """Test state representation for neural networks."""
        obs, info = self.env.reset()
        state = self.env.get_state_representation()
        
        self.assertEqual(len(state), 9)  # 3x3 grid
        self.assertEqual(state[0], 1.0)  # Position (0,0) should be 1
        self.assertEqual(sum(state), 1.0)  # Should be one-hot


class TestQLearningAgent(unittest.TestCase):
    """Test cases for QLearningAgent."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.agent = QLearningAgent(
            state_size=(3, 3),
            action_size=4,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1,
            seed=42,
        )

    def test_initialization(self) -> None:
        """Test agent initialization."""
        self.assertEqual(self.agent.state_size, (3, 3))
        self.assertEqual(self.agent.action_size, 4)
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.9)
        self.assertEqual(self.agent.epsilon, 0.1)
        self.assertEqual(self.agent.q_table.shape, (3, 3, 4))

    def test_get_action(self) -> None:
        """Test action selection."""
        state = np.array([0, 0])
        
        # Test exploitation (should return best action)
        action = self.agent.get_action(state, training=False)
        self.assertIn(action, range(4))
        
        # Test exploration (should return random action)
        actions = [self.agent.get_action(state, training=True) for _ in range(100)]
        self.assertTrue(len(set(actions)) > 1)  # Should have variety

    def test_update(self) -> None:
        """Test Q-table update."""
        state = np.array([0, 0])
        action = 1
        reward = 10.0
        next_state = np.array([1, 0])
        
        # Store original Q-value
        original_q = self.agent.q_table[0, 0, 1]
        
        # Update Q-table
        self.agent.update(state, action, reward, next_state, True)
        
        # Check that Q-value was updated
        new_q = self.agent.q_table[0, 0, 1]
        self.assertNotEqual(original_q, new_q)
        self.assertEqual(new_q, reward)  # Should equal reward for terminal state

    def test_decay_epsilon(self) -> None:
        """Test epsilon decay."""
        original_epsilon = self.agent.epsilon
        
        # Decay epsilon
        self.agent.decay_epsilon()
        
        # Check that epsilon decreased
        self.assertLess(self.agent.epsilon, original_epsilon)
        self.assertGreater(self.agent.epsilon, self.agent.epsilon_min)

    def test_save_load(self) -> None:
        """Test saving and loading agent."""
        # Train agent a bit
        env = create_simple_gridworld()
        self.agent.train_episode(env)
        
        # Save agent
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            self.agent.save(tmp.name)
            
            # Load agent
            loaded_agent = QLearningAgent.load(tmp.name)
            
            # Check that loaded agent matches original
            self.assertEqual(loaded_agent.state_size, self.agent.state_size)
            self.assertEqual(loaded_agent.action_size, self.agent.action_size)
            self.assertEqual(loaded_agent.learning_rate, self.agent.learning_rate)
            np.testing.assert_array_equal(loaded_agent.q_table, self.agent.q_table)
            
            # Clean up
            os.unlink(tmp.name)


class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for EvaluationMetrics."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.metrics = EvaluationMetrics()

    def test_add_episode(self) -> None:
        """Test adding episode data."""
        self.metrics.add_episode(
            episode=1,
            reward=10.0,
            length=5,
            success=True,
            algorithm="Test",
        )
        
        self.assertEqual(len(self.metrics.episode_data), 1)
        self.assertEqual(self.metrics.episode_data[0]["reward"], 10.0)

    def test_calculate_metrics(self) -> None:
        """Test metrics calculation."""
        # Add some test data
        for i in range(100):
            self.metrics.add_episode(
                episode=i,
                reward=10.0 if i < 50 else 5.0,
                length=10,
                success=i < 50,
                algorithm="Test",
            )
        
        metrics = self.metrics.calculate_metrics("Test")
        
        self.assertIn("mean_reward", metrics)
        self.assertIn("success_rate", metrics)
        self.assertIn("sample_efficiency", metrics)
        self.assertEqual(metrics["success_rate"], 0.5)  # 50% success rate

    def test_get_leaderboard(self) -> None:
        """Test leaderboard generation."""
        # Add data for multiple algorithms
        algorithms = ["Algorithm1", "Algorithm2"]
        for algo in algorithms:
            for i in range(50):
                self.metrics.add_episode(
                    episode=i,
                    reward=10.0 if algo == "Algorithm1" else 5.0,
                    length=10,
                    success=algo == "Algorithm1",
                    algorithm=algo,
                )
        
        leaderboard = self.metrics.get_leaderboard(algorithms)
        
        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(leaderboard.iloc[0]["Algorithm"], "Algorithm1")  # Should be first


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_set_seed(self) -> None:
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        np_rand1 = np.random.random()
        py_rand1 = np.random.random()
        
        # Reset seed and generate again
        set_seed(42)
        np_rand2 = np.random.random()
        py_rand2 = np.random.random()
        
        # Should be the same
        self.assertEqual(np_rand1, np_rand2)
        self.assertEqual(py_rand1, py_rand2)

    def test_get_device(self) -> None:
        """Test device detection."""
        device = get_device()
        self.assertIsNotNone(device)

    def test_timer(self) -> None:
        """Test timer context manager."""
        with Timer("Test Operation") as timer:
            import time
            time.sleep(0.1)
        
        self.assertIsNotNone(timer.elapsed)
        self.assertGreater(timer.elapsed, 0.05)

    def test_moving_average(self) -> None:
        """Test moving average calculation."""
        data = [1, 2, 3, 4, 5]
        window = 3
        
        result = moving_average(data, window)
        
        self.assertEqual(len(result), len(data))
        self.assertEqual(result[0], 1.0)  # First element unchanged
        self.assertEqual(result[2], 2.0)   # Average of [1, 2, 3]

    def test_confidence_interval(self) -> None:
        """Test confidence interval calculation."""
        data = [1, 2, 3, 4, 5]
        
        mean, lower, upper = calculate_confidence_interval(data)
        
        self.assertGreater(upper, mean)
        self.assertLess(lower, mean)
        self.assertGreater(mean, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def test_end_to_end_training(self) -> None:
        """Test complete training pipeline."""
        # Create environment
        env = create_simple_gridworld()
        
        # Create agent
        agent = QLearningAgent(
            state_size=env.grid_size,
            action_size=env.action_space.n,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1,
            seed=42,
        )
        
        # Train for a few episodes
        training_stats = agent.train(env, num_episodes=10, verbose=False)
        
        # Check that training produced results
        self.assertEqual(len(training_stats["episode_rewards"]), 10)
        self.assertEqual(len(training_stats["episode_lengths"]), 10)
        
        # Evaluate agent
        eval_stats = agent.evaluate(env, num_episodes=5)
        
        # Check evaluation results
        self.assertIn("mean_reward", eval_stats)
        self.assertIn("success_rate", eval_stats)
        self.assertGreaterEqual(eval_stats["success_rate"], 0.0)
        self.assertLessEqual(eval_stats["success_rate"], 1.0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
