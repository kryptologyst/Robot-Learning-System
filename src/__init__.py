"""Robot Learning System - Main Package.

A modern, research-focused robot learning system implementing reinforcement
learning algorithms for navigation and control tasks.
"""

from .algorithms import QLearningAgent
from .environments import GridWorld, create_default_gridworld, create_simple_gridworld
from .evaluation import EvaluationMetrics, compare_algorithms
from .utils import set_seed, get_device, setup_logging, Timer
from .visualization import LearningVisualizer, create_leaderboard_plot

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    # Algorithms
    "QLearningAgent",
    
    # Environments
    "GridWorld",
    "create_default_gridworld", 
    "create_simple_gridworld",
    
    # Evaluation
    "EvaluationMetrics",
    "compare_algorithms",
    
    # Utilities
    "set_seed",
    "get_device", 
    "setup_logging",
    "Timer",
    
    # Visualization
    "LearningVisualizer",
    "create_leaderboard_plot",
]
