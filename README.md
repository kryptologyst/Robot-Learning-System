# Robot Learning System

Research-focused robot learning system implementing reinforcement learning algorithms for navigation and control tasks. This project provides a comprehensive framework for robot learning with multiple RL algorithms, simulation environments, and evaluation metrics.

## Project Type: Learning - Reinforcement Learning (Off-policy)

This project focuses on robot learning through reinforcement learning, specifically implementing:
- Q-learning (tabular)
- Deep Q-Network (DQN)
- Policy Gradient methods (REINFORCE, PPO)
- Advanced RL techniques (Double DQN, Prioritized Experience Replay)

## Features

- **Multiple RL Algorithms**: Q-learning, DQN, Double DQN, PPO, REINFORCE
- **Simulation Environments**: Grid navigation, continuous control tasks
- **Modern Stack**: PyTorch, ROS 2, comprehensive logging
- **Evaluation Metrics**: Success rate, sample efficiency, learning curves
- **Visualization**: Interactive demos, trajectory plotting, performance analysis
- **Safety**: Simulation-first approach with comprehensive disclaimers

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.x
- ROS 2 Humble (optional, for advanced features)

### Installation

```bash
# Clone and setup
git clone https://github.com/kryptologyst/Robot-Learning-System.git
cd Robot-Learning-System

# Install dependencies
pip install -r requirements.txt

# For ROS 2 support (optional)
# Follow ROS 2 Humble installation guide
```

### Basic Usage

```bash
# Run basic Q-learning demo
python scripts/train_qlearning.py

# Run advanced RL comparison
python scripts/compare_algorithms.py

# Launch interactive demo
streamlit run demo/app.py
```

## Project Structure

```
src/
├── environments/          # Simulation environments
├── algorithms/           # RL algorithm implementations
├── utils/               # Utilities and helpers
├── evaluation/          # Metrics and evaluation tools
└── visualization/       # Plotting and visualization

config/                  # Configuration files
data/                   # Datasets and logs
scripts/                # Training and evaluation scripts
demo/                   # Interactive demos
tests/                  # Unit tests
assets/                 # Generated plots and videos
```

## Algorithms Implemented

1. **Q-Learning**: Tabular reinforcement learning for discrete state spaces
2. **Deep Q-Network (DQN)**: Deep RL for continuous state spaces
3. **Double DQN**: Improved DQN with reduced overestimation bias
4. **PPO**: Proximal Policy Optimization for continuous control
5. **REINFORCE**: Policy gradient method

## Environments

- **GridWorld**: Discrete navigation with obstacles
- **ContinuousNavigation**: Continuous state/action space
- **RobotArm**: 2-DOF arm reaching task

## Evaluation Metrics

- **Success Rate**: Percentage of successful episodes
- **Sample Efficiency**: Episodes to reach performance threshold
- **Learning Curves**: Reward progression over training
- **Stability**: Variance in performance across runs

## Safety and Limitations

⚠️ **IMPORTANT**: This project is for research and education only. See [DISCLAIMER.md](DISCLAIMER.md) for safety information.

- No real-time guarantees
- No hardware safety limits
- Simulation environments only
- Requires expert review for real robot deployment

## Contributing

1. Follow the code style (black + ruff)
2. Add type hints and docstrings
3. Include unit tests for new features
4. Update documentation

## License

MIT License - See LICENSE file for details.
# Robot-Learning-System
