"""Visualization tools for robot learning system.

This module provides comprehensive visualization capabilities including
learning curves, policy visualization, trajectory plotting, and interactive demos.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LearningVisualizer:
    """Visualization tools for learning algorithms."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')

    def plot_learning_curves(
        self,
        episode_rewards: List[float],
        episode_lengths: Optional[List[int]] = None,
        success_rates: Optional[List[float]] = None,
        window: int = 50,
        title: str = "Learning Curves",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot learning curves for training.

        Args:
            episode_rewards: List of episode rewards
            episode_lengths: Optional list of episode lengths
            success_rates: Optional list of success rates
            window: Window size for smoothing
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)

        # Plot rewards
        axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
        if len(episode_rewards) >= window:
            smoothed_rewards = self._moving_average(episode_rewards, window)
            axes[0, 0].plot(smoothed_rewards, color='blue', linewidth=2, label=f'Smoothed (window={window})')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot episode lengths
        if episode_lengths:
            axes[0, 1].plot(episode_lengths, alpha=0.3, color='green', label='Raw')
            if len(episode_lengths) >= window:
                smoothed_lengths = self._moving_average(episode_lengths, window)
                axes[0, 1].plot(smoothed_lengths, color='green', linewidth=2, label=f'Smoothed (window={window})')
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Plot success rates
        if success_rates:
            axes[1, 0].plot(success_rates, color='red', linewidth=2)
            axes[1, 0].set_title('Success Rate')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)

        # Plot reward distribution
        axes[1, 1].hist(episode_rewards, bins=30, alpha=0.7, color='purple')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curves to {save_path}")
        
        plt.show()

    def plot_policy(
        self,
        q_table: NDArray[np.float32],
        grid_size: Tuple[int, int],
        obstacles: Optional[List[Tuple[int, int]]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        goal_position: Optional[Tuple[int, int]] = None,
        title: str = "Learned Policy",
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize the learned policy.

        Args:
            q_table: Q-value table
            grid_size: Size of the grid
            obstacles: List of obstacle positions
            start_position: Starting position
            goal_position: Goal position
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create policy matrix
        policy = np.argmax(q_table, axis=2)
        
        # Create visualization
        height, width = grid_size
        
        # Plot arrows for policy
        for i in range(height):
            for j in range(width):
                action = policy[i, j]
                
                # Arrow directions: 0=up, 1=down, 2=left, 3=right
                arrow_dirs = {
                    0: (0, 0.3),   # up
                    1: (0, -0.3),  # down
                    2: (-0.3, 0),  # left
                    3: (0.3, 0),   # right
                }
                
                dx, dy = arrow_dirs[action]
                ax.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1, 
                        fc='black', ec='black', alpha=0.7)
        
        # Add obstacles
        if obstacles:
            for obs in obstacles:
                ax.add_patch(plt.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
                                         facecolor='black', alpha=0.8))
        
        # Add start position
        if start_position:
            ax.add_patch(plt.Circle((start_position[1], start_position[0]), 0.3, 
                                 facecolor='blue', alpha=0.8))
            ax.text(start_position[1], start_position[0], 'S', ha='center', va='center', 
                   color='white', fontweight='bold')
        
        # Add goal position
        if goal_position:
            ax.add_patch(plt.Circle((goal_position[1], goal_position[0]), 0.3, 
                                 facecolor='green', alpha=0.8))
            ax.text(goal_position[1], goal_position[0], 'G', ha='center', va='center', 
                   color='white', fontweight='bold')
        
        # Set up the plot
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert y-axis to match matrix indexing
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved policy plot to {save_path}")
        
        plt.show()

    def plot_q_values(
        self,
        q_table: NDArray[np.float32],
        grid_size: Tuple[int, int],
        title: str = "Q-Values Heatmap",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot Q-values as heatmap.

        Args:
            q_table: Q-value table
            grid_size: Size of the grid
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        action_names = ['Up', 'Down', 'Left', 'Right']
        
        for action in range(4):
            row, col = action // 2, action % 2
            
            # Extract Q-values for this action
            q_values = q_table[:, :, action]
            
            # Create heatmap
            im = axes[row, col].imshow(q_values, cmap='viridis', aspect='equal')
            axes[row, col].set_title(f'Q-Values for Action: {action_names[action]}')
            axes[row, col].set_xlabel('X Position')
            axes[row, col].set_ylabel('Y Position')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row, col])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Q-values heatmap to {save_path}")
        
        plt.show()

    def plot_trajectory(
        self,
        trajectory: List[Tuple[int, int]],
        grid_size: Tuple[int, int],
        obstacles: Optional[List[Tuple[int, int]]] = None,
        title: str = "Robot Trajectory",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot robot trajectory.

        Args:
            trajectory: List of (row, col) positions
            grid_size: Size of the grid
            obstacles: List of obstacle positions
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Extract coordinates
        if trajectory:
            rows, cols = zip(*trajectory)
            
            # Plot trajectory
            ax.plot(cols, rows, 'b-', linewidth=2, alpha=0.7, label='Path')
            ax.scatter(cols, rows, c='blue', s=50, alpha=0.7)
            
            # Mark start and end
            ax.scatter(cols[0], rows[0], c='green', s=200, marker='o', 
                      label='Start', edgecolors='black', linewidth=2)
            ax.scatter(cols[-1], rows[-1], c='red', s=200, marker='s', 
                      label='End', edgecolors='black', linewidth=2)
        
        # Add obstacles
        if obstacles:
            for obs in obstacles:
                ax.add_patch(plt.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
                                         facecolor='black', alpha=0.8))
        
        # Set up the plot
        height, width = grid_size
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trajectory plot to {save_path}")
        
        plt.show()

    def plot_algorithm_comparison(
        self,
        results: Dict[str, List[float]],
        metric: str = "reward",
        title: str = "Algorithm Comparison",
        save_path: Optional[str] = None,
    ) -> None:
        """Compare multiple algorithms.

        Args:
            results: Dictionary mapping algorithm names to metric values
            metric: Name of the metric being compared
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        data_for_box = [results[algo] for algo in results.keys()]
        labels = list(results.keys())
        
        ax1.boxplot(data_for_box, labels=labels)
        ax1.set_title(f'{metric.title()} Distribution')
        ax1.set_ylabel(metric.title())
        ax1.grid(True, alpha=0.3)
        
        # Bar plot with error bars
        means = [np.mean(results[algo]) for algo in labels]
        stds = [np.std(results[algo]) for algo in labels]
        
        bars = ax2.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
        ax2.set_title(f'{metric.title()} Comparison')
        ax2.set_ylabel(f'Mean {metric.title()}')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.2f}±{std:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved algorithm comparison to {save_path}")
        
        plt.show()

    def create_interactive_plot(
        self,
        episode_rewards: List[float],
        episode_lengths: Optional[List[int]] = None,
        title: str = "Interactive Learning Curves",
    ) -> go.Figure:
        """Create interactive plotly visualization.

        Args:
            episode_rewards: List of episode rewards
            episode_lengths: Optional list of episode lengths
            title: Plot title

        Returns:
            Plotly figure object
        """
        if episode_lengths:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Episode Rewards', 'Episode Lengths'),
                vertical_spacing=0.1
            )
            
            # Add rewards
            fig.add_trace(
                go.Scatter(
                    y=episode_rewards,
                    mode='lines',
                    name='Rewards',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add lengths
            fig.add_trace(
                go.Scatter(
                    y=episode_lengths,
                    mode='lines',
                    name='Lengths',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=episode_rewards,
                    mode='lines',
                    name='Rewards',
                    line=dict(color='blue', width=2)
                )
            )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig

    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average."""
        if len(data) < window:
            return data
        
        result = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i + 1]
            result.append(sum(window_data) / len(window_data))
        
        return result


def create_leaderboard_plot(
    leaderboard_data: Dict[str, Dict[str, float]],
    metric: str = "success_rate",
    title: str = "Algorithm Leaderboard",
    save_path: Optional[str] = None,
) -> None:
    """Create leaderboard visualization.

    Args:
        leaderboard_data: Dictionary mapping algorithm names to metrics
        metric: Primary metric for ranking
        title: Plot title
        save_path: Optional path to save plot
    """
    algorithms = list(leaderboard_data.keys())
    values = [leaderboard_data[algo][metric] for algo in algorithms]
    
    # Sort by metric value
    sorted_pairs = sorted(zip(algorithms, values), key=lambda x: x[1], reverse=True)
    sorted_algorithms, sorted_values = zip(*sorted_pairs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(sorted_algorithms, sorted_values, alpha=0.7)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, sorted_values):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved leaderboard plot to {save_path}")
    
    plt.show()
