import torch
import gymnasium as gym
import DQN as dqn
import os
from datetime import datetime
from gymnasium.wrappers import RecordVideo


class VideoRecorder:
    """Utility class for recording trained DQN policies as videos"""
    
    def __init__(self, device=None):
        """Initialize VideoRecorder with device
        
        Args:
            device: Torch device (cpu or cuda). If None, automatically selects.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
    
    def record_video(self, model_path, episodes=3, is_slippery=False, video_dir="videos"):
        """Record trained policy running in the environment as a video
        
        Args:
            model_path: Path to trained DQN model
            episodes: Number of episodes to record
            is_slippery: Whether the FrozenLake environment is slippery
            video_dir: Directory to save videos
        """
        # Create video directory if it doesn't exist
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        # Create FrozenLake environment
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode="rgb_array")
        
        # Wrap with RecordVideo to save videos
        timestamp = datetime.now().strftime("%Y_%m%d_%I_%M_%S_%p")
        video_path = os.path.join(video_dir, f"policy_recording_{timestamp}")
        env = RecordVideo(env, video_path, episode_trigger=lambda x: True)  # Record all episodes
        
        num_states = env.unwrapped.observation_space.n
        num_actions = env.unwrapped.action_space.n
        
        # Load trained policy
        policy_dqn = dqn.DQN(state_size=num_states, action_size=num_actions).to(self.device)
        policy_dqn.load_state_dict(torch.load(model_path, map_location=self.device))
        policy_dqn.eval()
        
        print(f"Recording policy from: {model_path}")
        print(f"Videos will be saved to: {video_path}")
        
        for ep in range(episodes):
            state, _ = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            
            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(self._encode_state(state, num_states)).argmax().item()
                
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
            
            print(f"Episode {ep+1}/{episodes} | Reward: {episode_reward}")
        
        env.close()
        print(f"\nVideo recording complete! Saved to: {video_path}")
