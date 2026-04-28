import torch 
import torch.nn as nn
import numpy as np
import gymnasium as gym
import DQN as dqn
import experience_replay as ExpMemory
import os
import matplotlib.pyplot as plt
from datetime import datetime
from video_recorder import VideoRecorder

class Agent(VideoRecorder):
    def __init__(self):
        super().__init__()  # Initialize VideoRecorder parent class
        self.epsilon = 1.0
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.0
        self.learning_rate = 0.001 
        self.discount_factor = 0.9
        self.target_sync_rate = 10

    def _encode_state(self, state, state_size):
        """Convert scalar state to one-hot encoded tensor for discrete environments"""
        one_hot = torch.zeros(state_size, dtype=torch.float).to(self.device)
        one_hot[int(state)] = 1
        return one_hot
    
    def run(self, envName, render, isSlippery, isTraining, episodes):
        print(f"Using device: {self.device}")
        env = gym.make(envName, map_name="4x4", is_slippery=isSlippery, max_episode_steps=200, render_mode="human" if render else "None")

        state_size = env.observation_space.n
        action_size = env.action_space.n
        rnd = np.random.default_rng()       
        
        if isTraining:
            policy_dqn = dqn.DQN(state_size, action_size).to(self.device)      
            target_dqn = dqn.DQN(state_size, action_size).to(self.device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.batch_size = 32
            self.epsilon_decay = 1/episodes

            self.rewards_per_episode = []
            self.epsilon_history = []      

            self.memory = ExpMemory.ReplayMemory(1000000)

            # Loss function and Optimizer for POlicy DQN vs Target DQN
            self.loss_fnc = nn.MSELoss()
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

            # Keep track of steps for target network synchronization
            step = 0
            
             # Create runs folder with timestamp
            runs_dir = "runs"
            if not os.path.exists(runs_dir):
                os.makedirs(runs_dir)
            
            timestamp = datetime.now().strftime("%Y_%m%d_%I_%M_%S_%p")
            run_dir = os.path.join(runs_dir, timestamp)
            os.makedirs(run_dir, exist_ok=True)
            
            log_file = os.path.join(run_dir, "training_log.txt")

            # Log training configuration
            with open(log_file, "w") as f:
                f.write(f"Training Configuration\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Environment: {envName}\n")
                f.write(f"Episodes: {episodes}\n")
                f.write(f"Learning Rate: {self.learning_rate}\n")
                f.write(f"Discount Factor: {self.discount_factor}\n")
                f.write(f"Epsilon Start: {self.epsilon}\n")
                f.write(f"Epsilon Decay: {self.epsilon_decay}\n")
                f.write(f"Epsilon Min: {self.epsilon_min}\n")
                f.write(f"Batch Size: {self.batch_size}\n")
                f.write(f"Target Sync Rate: {self.target_sync_rate}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"\n\nTraining Progress\n")
                f.write(f"=" * 50 + "\n")            

        for i in range(episodes):
            terminated = False
            exceeded = False
            episode_reward = 0      
            state,_ = env.reset()          

            while not terminated and not exceeded:
                # Epsilon-greedy action selection
                if isTraining and rnd.random() < self.epsilon:
                    action = env.action_space.sample()  # Random action                   
                else:
                    # Use the DQN to select an action
                    with torch.no_grad():
                        action = policy_dqn(self._encode_state(state, state_size)).argmax().item()

                new_state, reward, terminated, exceeded, _ = env.step(action)

                # Convert to tensors - one-hot encode the next state                
                # reward_tensor = torch.tensor(reward, dtype=torch.float).to(self.device)

                if isTraining:
                    self.memory.push((state, action, reward, new_state, terminated))                    
                    step += 1
                    state = new_state                    
                    episode_reward += reward             
            
            if isTraining:

                self.rewards_per_episode.append(float(episode_reward))       
           
                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
                self.epsilon_history.append(self.epsilon)

                # Print episode statistics
                avg_reward = np.mean(self.rewards_per_episode[-100:]) if len(self.rewards_per_episode) >= 100 else np.mean(self.rewards_per_episode)                
                print(f"Episode {i+1:4d}/{episodes} | Reward: {episode_reward:6.1f} | Avg Reward (100): {avg_reward:6.2f} | Epsilon: {self.epsilon:.4f}")
                
                # Log to file
                with open(log_file, "a") as f:
                    f.write(f"Episode {i+1:4d}/{episodes} | Reward: {episode_reward:6.1f} | Avg Reward (100): {avg_reward:6.2f} | Epsilon: {self.epsilon:.4f}\n")

                # Optimize the policy DQN using a batch of transitions from the replay memory
                if self.memory.len() > self.batch_size and np.sum(self.rewards_per_episode) > 0:
                    # Sample a batch of transitions from the replay memory
                    mini_batch = self.memory.sample(self.batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)                

                    if step >= self.target_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step = 0

        env.close()
        
        # Save Model Policy, Plots and Logs if training
        if isTraining:            
            torch.save(policy_dqn.state_dict(), os.path.join(run_dir, f"policy_dqn_{timestamp}.pt"))            
            self._save_training_plots(run_dir)
            print(f"\nTraining complete! Logs and plots saved to: {run_dir}")

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, model_path, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = dqn.DQN(state_size=num_states, action_size=num_actions).to(self.device)
        policy_dqn.load_state_dict(torch.load(model_path))
        policy_dqn.eval()    # switch model to evaluation mode    

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self._encode_state(state, num_states)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
           
        state, action, reward, next_state, terminated = zip(*mini_batch)

        # One-hot encode states and next states
        state = [self._encode_state(s, policy_dqn.fc1.in_features) for s in state]
        next_state = [self._encode_state(s, policy_dqn.fc1.in_features) for s in next_state]

        # Convert to tensors
        action = [torch.tensor(a, dtype=torch.long) for a in action]
        reward = [torch.tensor(r, dtype=torch.float) for r in reward]

        states = torch.stack(state)
        actions = torch.stack(action).to(self.device)
        rewards = torch.stack(reward).to(self.device)
        next_states = torch.stack(next_state)
        terminations = torch.tensor(terminated, dtype=torch.float).to(self.device)

        # calculating target Q-value using the Bellman equation
        # target_q = r + γ * max(Q(s', a')) if not terminal, else r
        # where: r = immediate reward
        #        γ = discount factor (0.99)
        #        Q(s', a') = Q-values of next states for all actions
        #        (1 - terminations) = masks out future rewards if episode ended
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor * target_dqn(next_states).max(dim=1)[0]

        current_policy_q = policy_dqn(states).gather(1, index=actions.unsqueeze(1)).squeeze()

        loss = self.loss_fnc(current_policy_q, target_q)

        self.optimizer.zero_grad() # Clear gradients before backward pass
        loss.backward()            # Backpropagate the loss to compute gradients
        self.optimizer.step()      # Update the policy network's parameters using the optimizer
        
    
    def _save_training_plots(self, run_dir):
        """Save training plots to the run directory"""
        
        # Plot rewards per episode
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Rewards plot
        axes[0].plot(self.rewards_per_episode, alpha=0.6, label='Episode Reward')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Rewards per Episode')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Add moving average
        if len(self.rewards_per_episode) > 100:
            moving_avg = np.convolve(self.rewards_per_episode, np.ones(100)/100, mode='valid')
            axes[0].plot(range(99, len(self.rewards_per_episode)), moving_avg, color='red', linewidth=2, label='100-Episode Moving Average')
            axes[0].legend()
                              
        # Epsilon decay plot
        axes[1].plot(self.epsilon_history, color='orange', alpha=0.6, label='Epsilon')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Epsilon')
        axes[1].set_title('Epsilon Decay')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(run_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        print(f"Plots saved to: {plot_path}")

if __name__ == "__main__":
    agent = Agent()
    input_mode = input("1. Train a new model\n2. Evaluate a trained model\n3. Record trained policy as video\nChoice: ").strip().lower()
    if input_mode == '1':
        episodes = input("Number of episodes to train (default=15000): ").strip()
        episodes = int(episodes) if episodes.isdigit() else 15000
        is_slippery_input = input("Is slippery? (y/n, default=y): ").strip().lower()
        is_slippery = is_slippery_input != 'n'
        agent.run(envName="FrozenLake-v1", render=False, isSlippery=is_slippery, isTraining=True, episodes=episodes)    
    elif input_mode == '2':
        model_path = input("Enter the path to the trained model (e.g., runs/2026_0426_03_45_22_PM/policy_dqn_2026_0426_03_45_22_PM.pt): ")
        episodes = input("Number of episodes to test (default=5): ").strip()
        episodes = int(episodes) if episodes.isdigit() else 5
        is_slippery_input = input("Is slippery? (y/n, default=y): ").strip().lower()
        is_slippery = is_slippery_input != 'n'
        agent.test(episodes=episodes, model_path=model_path, is_slippery=is_slippery)
    elif input_mode == '3':
        model_path = input("Enter the path to the trained model (e.g., runs/2026_0426_03_45_22_PM/policy_dqn_2026_0426_03_45_22_PM.pt): ")
        episodes = input("Number of episodes to record (default=3): ").strip()
        episodes = int(episodes) if episodes.isdigit() else 3
        is_slippery_input = input("Is slippery? (y/n, default=n): ").strip().lower()
        is_slippery = is_slippery_input == 'y'
        agent.record_video(model_path=model_path, episodes=episodes, is_slippery=is_slippery)