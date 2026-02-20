"""Evaluation metrics and leaderboard for robot learning system.

This module provides comprehensive evaluation metrics, statistical analysis,
and leaderboard functionality for comparing different RL algorithms.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for RL algorithms.

    Provides various metrics for evaluating RL algorithm performance including
    success rate, sample efficiency, learning curves, and statistical analysis.
    """

    def __init__(self) -> None:
        """Initialize evaluation metrics."""
        self.metrics: Dict[str, List[float]] = {}
        self.episode_data: List[Dict[str, Any]] = []

    def add_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        success: bool,
        algorithm: str,
        **kwargs: Any,
    ) -> None:
        """Add episode data for evaluation.

        Args:
            episode: Episode number
            reward: Episode reward
            length: Episode length
            success: Whether episode was successful
            algorithm: Algorithm name
            **kwargs: Additional episode data
        """
        episode_data = {
            "episode": episode,
            "reward": reward,
            "length": length,
            "success": success,
            "algorithm": algorithm,
            **kwargs,
        }
        self.episode_data.append(episode_data)

    def calculate_metrics(self, algorithm: str) -> Dict[str, float]:
        """Calculate comprehensive metrics for an algorithm.

        Args:
            algorithm: Algorithm name to calculate metrics for

        Returns:
            Dictionary of calculated metrics
        """
        # Filter data for specific algorithm
        algo_data = [ep for ep in self.episode_data if ep["algorithm"] == algorithm]
        
        if not algo_data:
            logger.warning(f"No data found for algorithm: {algorithm}")
            return {}

        rewards = [ep["reward"] for ep in algo_data]
        lengths = [ep["length"] for ep in algo_data]
        successes = [ep["success"] for ep in algo_data]

        metrics = {
            # Basic statistics
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "median_reward": np.median(rewards),
            
            # Episode length statistics
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            
            # Success metrics
            "success_rate": np.mean(successes),
            "total_episodes": len(algo_data),
            "successful_episodes": sum(successes),
            
            # Learning efficiency
            "sample_efficiency": self._calculate_sample_efficiency(rewards),
            "convergence_episode": self._find_convergence_episode(rewards),
        }

        # Add confidence intervals
        ci_mean, ci_lower, ci_upper = self._calculate_confidence_interval(rewards)
        metrics.update({
            "reward_ci_mean": ci_mean,
            "reward_ci_lower": ci_lower,
            "reward_ci_upper": ci_upper,
        })

        return metrics

    def _calculate_sample_efficiency(self, rewards: List[float], threshold: float = 0.8) -> int:
        """Calculate sample efficiency (episodes to reach threshold).

        Args:
            rewards: List of episode rewards
            threshold: Threshold for success (as fraction of max reward)

        Returns:
            Number of episodes to reach threshold
        """
        if not rewards:
            return 0
        
        max_reward = max(rewards)
        target_reward = threshold * max_reward
        
        # Calculate moving average
        window_size = min(50, len(rewards) // 10)
        if window_size < 5:
            window_size = 5
        
        moving_avg = self._moving_average(rewards, window_size)
        
        # Find first episode where moving average exceeds threshold
        for i, avg_reward in enumerate(moving_avg):
            if avg_reward >= target_reward:
                return i + window_size
        
        return len(rewards)  # Never reached threshold

    def _find_convergence_episode(self, rewards: List[float], window: int = 100) -> int:
        """Find episode where learning converges.

        Args:
            rewards: List of episode rewards
            window: Window size for convergence detection

        Returns:
            Episode number where convergence occurs
        """
        if len(rewards) < window:
            return len(rewards)
        
        # Calculate moving average and standard deviation
        moving_avg = self._moving_average(rewards, window)
        moving_std = self._moving_std(rewards, window)
        
        # Find where standard deviation becomes small relative to mean
        for i in range(window, len(rewards)):
            if moving_std[i - window] < 0.1 * abs(moving_avg[i - window]):
                return i
        
        return len(rewards)

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

    def _moving_std(self, data: List[float], window: int) -> List[float]:
        """Calculate moving standard deviation."""
        if len(data) < window:
            return [0.0] * len(data)
        
        result = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i + 1]
            result.append(np.std(window_data))
        
        return result

    def _calculate_confidence_interval(
        self, data: List[float], confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """Calculate confidence interval."""
        if not data:
            return 0.0, 0.0, 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # Use t-distribution for small samples
        if n < 30:
            from scipy import stats
            alpha = 1 - confidence
            t_value = stats.t.ppf(1 - alpha / 2, n - 1)
            margin_error = t_value * (std / np.sqrt(n))
        else:
            # Use normal distribution for large samples
            from scipy import stats
            alpha = 1 - confidence
            z_value = stats.norm.ppf(1 - alpha / 2)
            margin_error = z_value * (std / np.sqrt(n))
        
        return mean, mean - margin_error, mean + margin_error

    def get_leaderboard(self, algorithms: List[str]) -> pd.DataFrame:
        """Generate leaderboard comparing algorithms.

        Args:
            algorithms: List of algorithm names to compare

        Returns:
            DataFrame with algorithm comparison
        """
        leaderboard_data = []
        
        for algo in algorithms:
            metrics = self.calculate_metrics(algo)
            if metrics:
                leaderboard_data.append({
                    "Algorithm": algo,
                    "Success Rate": f"{metrics['success_rate']:.3f}",
                    "Mean Reward": f"{metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}",
                    "Sample Efficiency": metrics['sample_efficiency'],
                    "Convergence Episode": metrics['convergence_episode'],
                    "Total Episodes": metrics['total_episodes'],
                })
        
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by success rate (primary) and mean reward (secondary)
        df = df.sort_values(
            ["Success Rate", "Mean Reward"], 
            ascending=[False, False]
        ).reset_index(drop=True)
        
        return df

    def plot_learning_curves(
        self, 
        algorithms: List[str], 
        metric: str = "reward",
        window: int = 50
    ) -> None:
        """Plot learning curves for multiple algorithms.

        Args:
            algorithms: List of algorithm names
            metric: Metric to plot ('reward', 'length', 'success')
            window: Window size for smoothing
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        for algo in algorithms:
            algo_data = [ep for ep in self.episode_data if ep["algorithm"] == algo]
            if not algo_data:
                continue
            
            episodes = [ep["episode"] for ep in algo_data]
            values = [ep[metric] for ep in algo_data]
            
            # Calculate moving average
            if len(values) >= window:
                smoothed_values = self._moving_average(values, window)
                plt.plot(episodes[:len(smoothed_values)], smoothed_values, 
                        label=f"{algo} (smoothed)", linewidth=2)
            else:
                plt.plot(episodes, values, label=algo, alpha=0.7)
        
        plt.xlabel("Episode")
        plt.ylabel(metric.title())
        plt.title(f"Learning Curves - {metric.title()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_results(self, filepath: str) -> None:
        """Save evaluation results to file.

        Args:
            filepath: Path to save results
        """
        import json
        
        # Convert episode data to JSON-serializable format
        serializable_data = []
        for ep in self.episode_data:
            serializable_ep = {}
            for key, value in ep.items():
                if isinstance(value, np.ndarray):
                    serializable_ep[key] = value.tolist()
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_ep[key] = int(value)
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_ep[key] = float(value)
                else:
                    serializable_ep[key] = value
            serializable_data.append(serializable_ep)
        
        results = {
            "episode_data": serializable_data,
            "summary": self._generate_summary(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {filepath}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        algorithms = list(set(ep["algorithm"] for ep in self.episode_data))
        
        summary = {
            "total_episodes": len(self.episode_data),
            "algorithms": algorithms,
            "algorithm_metrics": {},
        }
        
        for algo in algorithms:
            summary["algorithm_metrics"][algo] = self.calculate_metrics(algo)
        
        return summary


def compare_algorithms(
    results: Dict[str, List[Dict[str, Any]]],
    metric: str = "reward"
) -> pd.DataFrame:
    """Compare multiple algorithms across runs.

    Args:
        results: Dictionary mapping algorithm names to list of run results
        metric: Metric to compare

    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    for algo_name, runs in results.items():
        all_values = []
        for run in runs:
            if metric in run:
                all_values.extend(run[metric])
        
        if all_values:
            comparison_data.append({
                "Algorithm": algo_name,
                "Mean": np.mean(all_values),
                "Std": np.std(all_values),
                "Min": np.min(all_values),
                "Max": np.max(all_values),
                "Median": np.median(all_values),
                "Runs": len(runs),
            })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("Mean", ascending=False).reset_index(drop=True)
    
    return df
