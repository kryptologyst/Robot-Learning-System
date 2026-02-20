"""Grid World environment for robot learning system.

This module implements a grid-based navigation environment for reinforcement learning,
supporting obstacles, multiple goals, and various reward structures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import Env, spaces
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class GridWorld(Env):
    """Grid World environment for robot navigation learning.

    A discrete grid environment where a robot must navigate from a start position
    to a goal position while avoiding obstacles.

    Args:
        grid_size: Size of the grid (height, width)
        start_position: Starting position (row, col)
        goal_position: Goal position (row, col)
        obstacles: List of obstacle positions [(row, col), ...]
        reward_structure: Reward configuration dict
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
    """

    # Action mapping: 0=up, 1=down, 2=left, 3=right
    ACTION_NAMES = ["up", "down", "left", "right"]
    ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(
        self,
        grid_size: Tuple[int, int] = (5, 5),
        start_position: Tuple[int, int] = (0, 0),
        goal_position: Tuple[int, int] = (4, 4),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        reward_structure: Optional[Dict[str, float]] = None,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.grid_size = grid_size
        self.start_position = np.array(start_position, dtype=np.int32)
        self.goal_position = np.array(goal_position, dtype=np.int32)
        self.obstacles = set(obstacles) if obstacles else set()
        self.max_steps = max_steps
        self.seed_value = seed

        # Default reward structure
        self.reward_structure = reward_structure or {
            "goal": 10.0,
            "step": -0.1,
            "obstacle": -1.0,
            "out_of_bounds": -1.0,
        }

        # State tracking
        self.position: NDArray[np.int32] = self.start_position.copy()
        self.step_count = 0
        self.episode_reward = 0.0

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=max(grid_size) - 1, shape=(2,), dtype=np.int32
        )

        # Set random seed
        self.reset(seed=seed)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[NDArray[np.int32], Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Initial observation and info dict
        """
        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)

        self.position = self.start_position.copy()
        self.step_count = 0
        self.episode_reward = 0.0

        info = {
            "position": self.position.copy(),
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
        }

        return self.position.copy(), info

    def step(self, action: int) -> Tuple[NDArray[np.int32], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to take (0-3)

        Returns:
            observation, reward, terminated, truncated, info
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Calculate new position
        delta = self.ACTION_DELTAS[action]
        new_position = self.position + np.array(delta, dtype=np.int32)

        # Check bounds
        if not self._is_valid_position(new_position):
            reward = self.reward_structure["out_of_bounds"]
            terminated = False
            truncated = True
        else:
            # Check obstacles
            if tuple(new_position) in self.obstacles:
                reward = self.reward_structure["obstacle"]
                terminated = False
                truncated = True
            else:
                # Move to new position
                self.position = new_position
                self.step_count += 1

                # Check if goal reached
                if np.array_equal(self.position, self.goal_position):
                    reward = self.reward_structure["goal"]
                    terminated = True
                    truncated = False
                else:
                    reward = self.reward_structure["step"]
                    terminated = False
                    truncated = self.step_count >= self.max_steps

        self.episode_reward += reward

        info = {
            "position": self.position.copy(),
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
            "action": action,
            "action_name": self.ACTION_NAMES[action],
        }

        return self.position.copy(), reward, terminated, truncated, info

    def _is_valid_position(self, position: NDArray[np.int32]) -> bool:
        """Check if position is within grid bounds.

        Args:
            position: Position to check

        Returns:
            True if position is valid
        """
        return (
            0 <= position[0] < self.grid_size[0]
            and 0 <= position[1] < self.grid_size[1]
        )

    def get_possible_actions(self) -> List[int]:
        """Get list of valid actions from current position.

        Returns:
            List of valid action indices
        """
        valid_actions = []
        for action, delta in enumerate(self.ACTION_DELTAS):
            new_position = self.position + np.array(delta, dtype=np.int32)
            if (
                self._is_valid_position(new_position)
                and tuple(new_position) not in self.obstacles
            ):
                valid_actions.append(action)
        return valid_actions

    def render(self, mode: str = "human") -> Optional[NDArray[np.uint8]]:
        """Render the environment.

        Args:
            mode: Rendering mode ('human' or 'rgb_array')

        Returns:
            Rendered image if mode is 'rgb_array', None otherwise
        """
        if mode == "human":
            self._render_console()
        elif mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def _render_console(self) -> None:
        """Render environment to console."""
        print(f"\nStep {self.step_count}, Position: {self.position}")
        print(f"Episode Reward: {self.episode_reward:.2f}")
        
        # Create grid representation
        grid = np.full(self.grid_size, ".")
        
        # Add obstacles
        for obs in self.obstacles:
            grid[obs] = "X"
        
        # Add goal
        grid[tuple(self.goal_position)] = "G"
        
        # Add robot
        grid[tuple(self.position)] = "R"
        
        # Print grid
        for row in grid:
            print(" ".join(row))
        print()

    def _render_rgb_array(self) -> NDArray[np.uint8]:
        """Render environment as RGB array.

        Returns:
            RGB image array
        """
        # Simple RGB rendering - can be enhanced
        height, width = self.grid_size
        img = np.zeros((height * 20, width * 20, 3), dtype=np.uint8)
        
        # Fill with background color
        img.fill(255)
        
        # Add obstacles (black)
        for obs in self.obstacles:
            y, x = obs
            img[y*20:(y+1)*20, x*20:(x+1)*20] = [0, 0, 0]
        
        # Add goal (green)
        gy, gx = self.goal_position
        img[gy*20:(gy+1)*20, gx*20:(gx+1)*20] = [0, 255, 0]
        
        # Add robot (red)
        ry, rx = self.position
        img[ry*20:(ry+1)*20, rx*20:(rx+1)*20] = [255, 0, 0]
        
        return img

    def get_state_representation(self) -> NDArray[np.float32]:
        """Get state representation for neural networks.

        Returns:
            State vector suitable for neural network input
        """
        # One-hot encoding of position
        state = np.zeros(self.grid_size[0] * self.grid_size[1], dtype=np.float32)
        idx = self.position[0] * self.grid_size[1] + self.position[1]
        state[idx] = 1.0
        return state

    def close(self) -> None:
        """Clean up environment resources."""
        pass


def create_default_gridworld() -> GridWorld:
    """Create a default GridWorld environment.

    Returns:
        Configured GridWorld environment
    """
    obstacles = [(1, 1), (2, 2), (3, 1)]
    return GridWorld(
        grid_size=(5, 5),
        start_position=(0, 0),
        goal_position=(4, 4),
        obstacles=obstacles,
        max_steps=50,
    )


def create_simple_gridworld() -> GridWorld:
    """Create a simple GridWorld environment without obstacles.

    Returns:
        Simple GridWorld environment
    """
    return GridWorld(
        grid_size=(5, 5),
        start_position=(0, 0),
        goal_position=(4, 4),
        obstacles=None,
        max_steps=20,
    )
