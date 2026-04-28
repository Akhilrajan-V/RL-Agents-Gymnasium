from collections import deque
import random

class ReplayMemory:
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)        
    
    def len(self):
        return len(self.memory)