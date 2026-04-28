# Gym Projects - Reinforcement Learning Algorithms

A collection of reinforcement learning implementations using OpenAI Gymnasium, exploring different RL algorithms applied to the **FrozenLake-v1** environment.

## Project Overview

This project implements and compares multiple reinforcement learning algorithms:
- **Q-Learning**: A model-free, value-based RL algorithm using tabular Q-tables
- **Deep Q-Networks (DQN)**: A neural network-based approach combining Q-learning with deep learning

Both algorithms are applied to solve the FrozenLake environment, where an agent learns to navigate a frozen lake grid while avoiding falling into holes.

---

## Folder Structure

```
Gym Projects/
├── README.md                          # Project documentation
├── Agent.py                           # DQN agent class with training/testing logic
├── DQN.py                             # Deep Q-Network neural network architecture
├── Q_Learning.py                      # Q-Learning algorithm implementation
├── experience_replay.py               # Experience replay memory for DQN training
├── validate_gym.py                    # Gymnasium environment validation script
├── runs/                              # Training outputs and models
│   └── QLearning/                     # Q-Learning models and results
│       ├── q_table_*.pkl              # Trained Q-tables
│       ├── training_plot_*.png        # Training performance plots
│       └── test_plot_*.png            # Testing performance plots
└── __pycache__/                       # Python cache files (auto-generated)
```

---

## Environment Setup

### Prerequisites
- Python 3.10 or higher
- Conda (Miniconda or Anaconda)

### Installation

#### Step 1: Create a Conda Environment

```bash
conda create -n gym-rl python=3.10
conda activate gym-rl
```

#### Step 2: Install Required Packages

```bash
# Core dependencies
conda install -c pytorch pytorch::pytorch pytorch::pytorch-cuda=11.8 -y
conda install -c conda-forge gymnasium matplotlib numpy scikit-learn -y

# Alternative: Using pip
pip install gymnasium torch matplotlib numpy
```

#### Full Requirements

**File: `requirements.txt`** (if using pip)
```
gymnasium>=0.27.0
torch>=2.0.0
numpy>=1.23.0
matplotlib>=3.6.0
```

To create and install from requirements.txt:
```bash
pip install -r requirements.txt
```

---

## File Descriptions

### Core Algorithm Files

| File | Description |
|------|-------------|
| `Q_Learning.py` | Implements tabular Q-Learning with epsilon-greedy exploration. Features configurable training/testing modes, slippery environment support, and automatic model saving to `runs/QLearning/` |
| `Agent.py` | DQN agent class handling training and testing loops. Uses neural networks to approximate Q-values, implements epsilon-decay, target network syncing, and replay memory sampling |
| `DQN.py` | PyTorch neural network architecture for value function approximation. Simple fully-connected network with ReLU activation |
| `experience_replay.py` | Replay memory buffer for storing and sampling experiences. Enables decorrelated training data sampling |

### Utility Files

| File | Description |
|------|-------------|
| `validate_gym.py` | Simple script to verify Gymnasium installation and environment rendering |

---

## Usage

### Q-Learning

Run interactive training or testing with Q-Learning:

```bash
python Q_Learning.py
```

**Interactive Prompts:**
- **Mode**: Select `train` to train a new model or `test` to evaluate an existing model
- **Slippery**: Choose `yes` for slippery ice (stochastic) or `no` for deterministic environment
- **Episodes**: (Training only) Specify the number of training episodes

**Example Session:**
```
==================================================
Q-Learning FrozenLake Training/Testing
==================================================

Select mode (train/test): train
Use slippery environment? (yes/no): yes
Number of training episodes: 10000

Starting training with isSlippery=True...
Episode: 100/10000 | Avg Reward (last 100): 0.12 | Avg Steps: 156.4 | Epsilon: 0.9985 | Learning Rate: 0.9000
...
Training Complete!
Total Episodes: 10000
Final Epsilon: 0.0000
Average Reward (last 100 episodes): 0.45
Best Episode Reward: 1
Model saved to: runs/QLearning/q_table_10000ep_slipTrue.pkl
Plot saved as 'runs/QLearning/training_plot_10000ep_slipTrue.png'
```

### Testing a Trained Model

```bash
python Q_Learning.py
```

**Interactive Prompts:**
```
Select mode (train/test): test
Use slippery environment? (yes/no): yes
```

The system automatically loads the most recently trained model and runs evaluation episodes with visualization enabled.

---

## Hyperparameters

### Q-Learning

```python
learning_rate = 0.9           # Learning rate (α)
discount_factor = 0.8         # Discount factor (γ)
epsilon = 1.0                 # Initial exploration rate
epsilon_decay = 0.00001       # Epsilon decay per episode
```

### DQN (in Agent.py)

```python
learning_rate = 0.001         # Network learning rate
discount_factor = 0.9         # Discount factor (γ)
epsilon = 1.0                 # Initial exploration rate
epsilon_decay = 0.001         # Epsilon decay
target_sync_rate = 10         # Sync target network every N episodes
```

---

## DQN Demonstrations

### DQN Policy in Action

**Deterministic Environment (is_slippery=False)**
![DQN Video - Deterministic](media/DQN-video-episode.gif)

**Stochastic Environment (is_slippery=True)**
![DQN Video - Slippery](media/DQN-video-episode-slippery.gif)

These demonstrations show the trained DQN policy successfully navigating the FrozenLake environment. The agent learns to take optimal actions to reach the goal while avoiding holes.

---

## Output Structure

### Training Outputs

When training completes, the following files are saved to `runs/QLearning/`:

1. **Model File**: `q_table_{episodes}ep_slip{isSlippery}.pkl`
   - Serialized Q-table or DQN weights
   - Loadable for future testing/evaluation

2. **Training Plot**: `training_plot_{episodes}ep_slip{isSlippery}.png`
   - Episode rewards over time
   - Step count per episode
   - Moving average trends (100-episode window)

3. **Test Plot**: `test_plot_slip{isSlippery}.png`
   - Generated when testing a trained model

### Metrics Tracked

- **Episode Reward**: Total reward accumulated per episode
- **Episode Length**: Number of steps taken before termination
- **Epsilon**: Current exploration rate
- **Learning Rate**: Adaptive learning rate (decreases with training)

---

## Environment Details

### FrozenLake-v1

- **Grid Size**: 8x8 (64 states)
- **Action Space**: 4 discrete actions (Up, Down, Left, Right)
- **Reward**: +1 for reaching goal, 0 otherwise
- **Modes**:
  - `is_slippery=True`: Stochastic transitions (realistic)
  - `is_slippery=False`: Deterministic transitions (easier)

---

## Key Features

✅ **Interactive CLI**: User-friendly prompts for mode selection  
✅ **Automatic Model Management**: Models saved and loaded from organized directory structure  
✅ **Visualization**: Automatic plot generation for training/testing metrics  
✅ **Flexible Environments**: Support for both slippery and deterministic variants  
✅ **Adaptive Learning**: Epsilon decay and learning rate scheduling  
✅ **Performance Tracking**: Detailed logging of rewards, steps, and exploration metrics  

---

## Requirements

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| gymnasium | ≥0.27.0 | RL environment API |
| torch | ≥2.0.0 | Deep learning (DQN only) |
| numpy | ≥1.23.0 | Numerical computations |
| matplotlib | ≥3.6.0 | Visualization |
| pickle | Built-in | Model serialization |

### System Requirements

- **RAM**: 2GB minimum (4GB+ recommended)
- **GPU**: Optional (PyTorch will use CPU if unavailable)
- **Storage**: ~100MB for outputs (models + plots)

---

## Future Enhancements

- [ ] Policy Gradient methods (REINFORCE, Actor-Critic)
- [ ] Double DQN for reduced overestimation
- [ ] Dueling DQN architecture
- [ ] Different environment variants (CartPole, MountainCar)
- [ ] Hyperparameter tuning utilities
- [ ] Training convergence analysis
- [ ] Web-based visualization dashboard

---

## Troubleshooting

### Issue: "No module named 'gymnasium'"
```bash
conda install -c conda-forge gymnasium
```

### Issue: "CUDA not available" (if using GPU)
```bash
# Reinstall PyTorch for CPU only
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch -y
```

### Issue: "No saved model found in runs/QLearning/"
- Train a model first in `train` mode before testing
- Check that `runs/QLearning/` directory has `.pkl` files

### Issue: Plots not generating
```bash
# Ensure matplotlib is installed
conda install matplotlib
```

---

## License

This project is provided as-is for educational purposes.

---

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Deep Q-Networks (DQN) Paper](https://www.nature.com/articles/nature14236)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

---

## Author Notes

This project serves as a learning resource for understanding fundamental RL algorithms and their implementations. The modular structure allows for easy extension with new algorithms or environments.
