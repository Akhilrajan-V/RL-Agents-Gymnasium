import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path

def get_model_dir():
    """Create and return the model directory path"""
    model_dir = Path("runs/QLearning")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def find_latest_model(model_dir):
    """Find the most recently modified model file in the directory"""
    pkl_files = list(model_dir.glob("*.pkl"))
    if not pkl_files:
        return None
    return max(pkl_files, key=os.path.getmtime)

def QLearning(isSlippery, render, episodes, isTraining):    
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=isSlippery, max_episode_steps=200, render_mode="human" if render else "None")

    model_dir = get_model_dir()
    
    if isTraining:
        #initialize Q-table
        Q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        latest_model = find_latest_model(model_dir)
        if latest_model:
            with open(latest_model, 'rb') as f:
                Q_table = pickle.load(f)
            print(f"Loaded model: {latest_model.name}")
        else:
            raise FileNotFoundError("No saved model found in runs/QLearning/. Please train a model first.")

    # Hyperparameters
    learning_rate = 0.9  # Learning rate
    discount_factor = 0.8  # Discount factor
    epsilon = 1.0 # Exploration rate
    epsilon_decay = 0.00001  # Decay rate for exploration
    rnd = np.random.default_rng() 

    # Track rewards for each episode
    episode_rewards = []
    episode_lengths = []

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        exceeded = False
        episode_reward = 0
        steps = 0

        while not terminated and not exceeded: 
            if isTraining and rnd.random() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(Q_table[state,:])  # Greedy action

            new_state, reward, terminated, exceeded, metaInfo = env.step(action)
            episode_reward += reward
            steps += 1
            
            if isTraining:
                Q_table[state, action] = Q_table[state, action] + learning_rate * (reward + discount_factor * np.max(Q_table[new_state,:]) - Q_table[state, action])
            
            state = new_state

        # decay epsilon
        epsilon = max(0, epsilon - epsilon_decay) 
    
        if epsilon == 0:
            learning_rate = 0.001    

        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Display metadata every 100 episodes
        if (i + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_lengths[-100:])
            print(f"Episode: {i + 1}/{episodes} | Avg Reward (last 100): {avg_reward:.2f} | Avg Steps: {avg_steps:.1f} | Epsilon: {epsilon:.4f} | Learning Rate: {learning_rate:.4f}")
    
    env.close()

    print(f"\nTraining Complete!")
    print(f"Total Episodes: {episodes}")
    print(f"Final Epsilon: {epsilon:.4f}")
    print(f"Average Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best Episode Reward: {max(episode_rewards)}")

    # Determine plot save path
    if isTraining:
        plot_filename = f"training_plot_{episodes}ep_slip{isSlippery}.png"
    else:
        plot_filename = f"test_plot_slip{isSlippery}.png"
    plot_save_path = model_dir / plot_filename

    # Plot episodes vs rewards
    plot_results(episode_rewards, episode_lengths, plot_save_path)

    if isTraining:
        model_filename = f"q_table_{episodes}ep_slip{isSlippery}.pkl"
        model_path = model_dir / model_filename
        with open(model_path, 'wb') as f:
            pickle.dump(Q_table, f)
        print(f"Model saved to: {model_path}")
    
    # return Q_table, episode_rewards, episode_lengths


def plot_results(episode_rewards, episode_lengths, save_path=None):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Episodes vs Rewards
    ax1.plot(episode_rewards, alpha=0.6, label='Reward per episode')
    
    # Add moving average with dynamic window size
    window = min(100, len(episode_rewards) // 2) if len(episode_rewards) > 1 else 1
    if window > 1 and len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Average')
    
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel('Reward')
    ax1.set_title('Q-Learning: Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episodes vs Step Count
    ax2.plot(episode_lengths, alpha=0.6, label='Steps per episode', color='green')
    
    if window > 1 and len(episode_lengths) >= window:
        moving_avg_steps = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), moving_avg_steps, color='orange', linewidth=2, label=f'{window}-Episode Moving Average')
    
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Number of Steps')
    ax2.set_title('Q-Learning: Episode Length Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = Path('training_results.png')
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved as '{save_path}'")
    # plt.show()

if __name__ == "__main__":
    # Get user input for mode
    print("=" * 50)
    print("Q-Learning FrozenLake Training/Testing")
    print("=" * 50)
    
    while True:
        mode = input("\nSelect mode (train/test): ").strip().lower()
        if mode in ['train', 'test', 't']:
            break
        print("Invalid input. Please enter 'train' or 'test'.")
    
    isTraining = mode.startswith('t') and mode != 'test'
    
    while True:
        slippery_input = input("Use slippery environment? (yes/no): ").strip().lower()
        if slippery_input in ['yes', 'y', 'no', 'n']:
            break
        print("Invalid input. Please enter 'yes' or 'no'.")
    
    isSlippery = slippery_input in ['yes', 'y']
    
    # Set episodes based on mode
    if isTraining:
        while True:
            try:
                episodes = int(input("Number of training episodes: ").strip())
                if episodes > 0:
                    break
                print("Episodes must be greater than 0.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        episodes = 1
    
    # Run QLearning
    print(f"\nStarting {'training' if isTraining else 'testing'} with isSlippery={isSlippery}...")
    render = not isTraining  # Render when testing, not when training
    QLearning(isSlippery=isSlippery, render=render, episodes=episodes, isTraining=isTraining)

