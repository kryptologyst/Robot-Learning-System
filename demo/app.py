"""Interactive Streamlit demo for Robot Learning System.

This demo provides an interactive interface for training and visualizing
Q-Learning agents on grid world environments.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms.qlearning import QLearningAgent
from environments.gridworld import GridWorld, create_default_gridworld, create_simple_gridworld
from evaluation import EvaluationMetrics
from utils import set_seed
from visualization import LearningVisualizer


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Robot Learning System Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Robot Learning System Demo")
    st.markdown("Interactive Q-Learning for Robot Navigation")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_type = st.sidebar.selectbox(
        "Environment Type",
        ["Simple", "Default"],
        help="Simple: No obstacles, Default: With obstacles"
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    num_episodes = st.sidebar.slider("Training Episodes", 100, 2000, 1000)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
    discount_factor = st.sidebar.slider("Discount Factor", 0.5, 0.99, 0.9, 0.01)
    epsilon = st.sidebar.slider("Initial Epsilon", 0.01, 0.5, 0.1, 0.01)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.99, 0.999, 0.995, 0.001)
    
    # Random seed
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=1000)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Training Progress")
        
        if st.button("🚀 Start Training", type="primary"):
            # Set random seed
            set_seed(seed)
            
            # Create environment
            if env_type == "Simple":
                env = create_simple_gridworld()
            else:
                env = create_default_gridworld()
            
            # Create agent
            agent = QLearningAgent(
                state_size=env.grid_size,
                action_size=env.action_space.n,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=0.01,
                seed=seed,
            )
            
            # Training progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training loop with progress updates
            episode_rewards = []
            episode_lengths = []
            success_rates = []
            
            for episode in range(num_episodes):
                # Train one episode
                episode_stats = agent.train_episode(env)
                
                episode_rewards.append(episode_stats["reward"])
                episode_lengths.append(episode_stats["length"])
                
                # Calculate success rate
                if episode >= 99:
                    recent_rewards = episode_rewards[-100:]
                    recent_successes = sum(1 for r in recent_rewards if r > 0)
                    success_rate = recent_successes / min(100, episode + 1)
                else:
                    success_rate = 0.0
                success_rates.append(success_rate)
                
                # Update progress
                progress = (episode + 1) / num_episodes
                progress_bar.progress(progress)
                
                if episode % 100 == 0:
                    status_text.text(f"Episode {episode + 1}/{num_episodes} - "
                                   f"Reward: {episode_stats['reward']:.2f}, "
                                   f"Success Rate: {success_rate:.2f}")
            
            # Store results in session state
            st.session_state.agent = agent
            st.session_state.env = env
            st.session_state.episode_rewards = episode_rewards
            st.session_state.episode_lengths = episode_lengths
            st.session_state.success_rates = success_rates
            
            st.success("Training completed!")
    
    with col2:
        st.header("Environment Info")
        
        if env_type == "Simple":
            st.info("**Simple Environment**\n\n- 5x5 grid\n- No obstacles\n- Start: (0,0)\n- Goal: (4,4)")
        else:
            st.info("**Default Environment**\n\n- 5x5 grid\n- Obstacles at (1,1), (2,2), (3,1)\n- Start: (0,0)\n- Goal: (4,4)")
        
        st.subheader("Training Parameters")
        st.write(f"- Episodes: {num_episodes}")
        st.write(f"- Learning Rate: {learning_rate}")
        st.write(f"- Discount Factor: {discount_factor}")
        st.write(f"- Epsilon: {epsilon}")
        st.write(f"- Epsilon Decay: {epsilon_decay}")
    
    # Results section
    if hasattr(st.session_state, 'agent'):
        st.header("📊 Results")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Learning Curves", "Policy", "Q-Values", "Evaluation"])
        
        with tab1:
            st.subheader("Learning Progress")
            
            # Create interactive plotly charts
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Episode Rewards', 'Episode Lengths', 'Success Rate', 'Reward Distribution'),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Episode rewards
            fig.add_trace(
                go.Scatter(
                    y=st.session_state.episode_rewards,
                    mode='lines',
                    name='Rewards',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            # Episode lengths
            fig.add_trace(
                go.Scatter(
                    y=st.session_state.episode_lengths,
                    mode='lines',
                    name='Lengths',
                    line=dict(color='green', width=1)
                ),
                row=1, col=2
            )
            
            # Success rate
            fig.add_trace(
                go.Scatter(
                    y=st.session_state.success_rates,
                    mode='lines',
                    name='Success Rate',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            
            # Reward distribution
            fig.add_trace(
                go.Histogram(
                    x=st.session_state.episode_rewards,
                    name='Reward Distribution',
                    nbinsx=30
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Learned Policy")
            
            # Create policy visualization
            q_table = st.session_state.agent.get_q_values()
            policy = np.argmax(q_table, axis=2)
            
            # Create grid visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            
            height, width = st.session_state.env.grid_size
            
            # Plot arrows for policy
            action_dirs = {
                0: (0, 0.3),   # up
                1: (0, -0.3),  # down
                2: (-0.3, 0),  # left
                3: (0.3, 0),   # right
            }
            
            for i in range(height):
                for j in range(width):
                    action = policy[i, j]
                    dx, dy = action_dirs[action]
                    ax.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1, 
                            fc='black', ec='black', alpha=0.7)
            
            # Add obstacles
            if hasattr(st.session_state.env, 'obstacles'):
                for obs in st.session_state.env.obstacles:
                    ax.add_patch(plt.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
                                             facecolor='black', alpha=0.8))
            
            # Add start and goal
            ax.add_patch(plt.Circle((st.session_state.env.start_position[1], 
                                   st.session_state.env.start_position[0]), 0.3, 
                                 facecolor='blue', alpha=0.8))
            ax.text(st.session_state.env.start_position[1], 
                   st.session_state.env.start_position[0], 'S', 
                   ha='center', va='center', color='white', fontweight='bold')
            
            ax.add_patch(plt.Circle((st.session_state.env.goal_position[1], 
                                   st.session_state.env.goal_position[0]), 0.3, 
                                 facecolor='green', alpha=0.8))
            ax.text(st.session_state.env.goal_position[1], 
                   st.session_state.env.goal_position[0], 'G', 
                   ha='center', va='center', color='white', fontweight='bold')
            
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(-0.5, height - 0.5)
            ax.set_aspect('equal')
            ax.set_title('Learned Policy')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
            
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Q-Values Heatmap")
            
            q_table = st.session_state.agent.get_q_values()
            action_names = ['Up', 'Down', 'Left', 'Right']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Q-Values Heatmap', fontsize=16)
            
            for action in range(4):
                row, col = action // 2, action % 2
                q_values = q_table[:, :, action]
                
                im = axes[row, col].imshow(q_values, cmap='viridis', aspect='equal')
                axes[row, col].set_title(f'Q-Values for Action: {action_names[action]}')
                axes[row, col].set_xlabel('X Position')
                axes[row, col].set_ylabel('Y Position')
                
                plt.colorbar(im, ax=axes[row, col])
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab4:
            st.subheader("Performance Metrics")
            
            # Calculate final metrics
            final_rewards = st.session_state.episode_rewards[-100:]
            final_success_rate = st.session_state.success_rates[-1]
            mean_reward = np.mean(final_rewards)
            std_reward = np.std(final_rewards)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Final Success Rate", f"{final_success_rate:.3f}")
            
            with col2:
                st.metric("Mean Reward (Last 100)", f"{mean_reward:.2f}")
            
            with col3:
                st.metric("Reward Std Dev", f"{std_reward:.2f}")
            
            # Additional statistics
            st.subheader("Training Statistics")
            
            stats_data = {
                "Metric": ["Total Episodes", "Final Epsilon", "Max Reward", "Min Reward", "Mean Episode Length"],
                "Value": [
                    len(st.session_state.episode_rewards),
                    f"{st.session_state.agent.epsilon:.3f}",
                    f"{max(st.session_state.episode_rewards):.2f}",
                    f"{min(st.session_state.episode_rewards):.2f}",
                    f"{np.mean(st.session_state.episode_lengths):.1f}"
                ]
            }
            
            st.table(stats_data)
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer**: This is a research/educational demo. Do not use on real robots without expert review and safety measures.")
    st.markdown("See [DISCLAIMER.md](DISCLAIMER.md) for safety information.")


if __name__ == "__main__":
    main()
