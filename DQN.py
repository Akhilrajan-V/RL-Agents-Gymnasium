import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=16):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_dim)
        # self.hd1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        # x = torch.nn.functional.relu(self.hd1(x))
        return self.fc2(x) 