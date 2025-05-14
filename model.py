import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import numpy as np

class DQNetwork(nn.Module):
    """Deep Q-Network with dual output branches for road and upgrade selection."""
    def __init__(self, state_size, num_road_segments, num_upgrade_types, hidden_size=128):
        super(DQNetwork, self).__init__()
        self.num_road_segments = num_road_segments
        self.num_upgrade_types = num_upgrade_types

        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Branch 1: Q-values for road segment selection
        self.road_adv = nn.Linear(hidden_size, num_road_segments)
        self.road_val = nn.Linear(hidden_size, 1)

        # Branch 2: Q-values for upgrade type selection
        self.upgrade_adv = nn.Linear(hidden_size, num_upgrade_types)
        self.upgrade_val = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Road selection branch (Dueling DQN)
        road_advantages = self.road_adv(x)
        road_value = self.road_val(x)
        # Q_road(s,a_road) = V_road(s) + (A_road(s,a_road) - mean(A_road(s,a_road)))
        q_values_road = road_value + (road_advantages - road_advantages.mean(dim=-1, keepdim=True))

        # Upgrade type selection branch (Dueling DQN)
        upgrade_advantages = self.upgrade_adv(x)
        upgrade_value = self.upgrade_val(x)
        # Q_upgrade(s,a_upgrade) = V_upgrade(s) + (A_upgrade(s,a_upgrade) - mean(A_upgrade(s,a_upgrade)))
        q_values_upgrade = upgrade_value + (upgrade_advantages - upgrade_advantages.mean(dim=-1, keepdim=True))
        
        return q_values_road, q_values_upgrade

Transition = namedtuple("Transition", 
                        ("state", "action_road", "action_upgrade", "next_state", "reward", "done"))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    # Example usage (for testing the components)
    # These dimensions need to match the environment
    # From environment.py: obs_space shape is (11,) for the sample data (9 roads + budget + timestep)
    # Action space: MultiDiscrete([num_road_segments, num_upgrade_types]) -> num_road_segments=9, num_upgrade_types=3
    state_dim = 11 
    n_roads = 9
    n_upgrades = 3

    # Test DQNetwork
    net = DQNetwork(state_dim, n_roads, n_upgrades)
    print("DQNetwork initialized:", net)
    # Create a dummy state tensor (batch_size=1)
    dummy_state = torch.randn(1, state_dim)
    q_roads, q_upgrades = net(dummy_state)
    print("Q-values for roads (shape):", q_roads.shape) # Expected: (1, n_roads)
    print("Q-values for upgrades (shape):", q_upgrades.shape) # Expected: (1, n_upgrades)

    # Test ReplayBuffer
    buffer = ReplayBuffer(1000)
    print("ReplayBuffer initialized with capacity 1000.")
    # Add some dummy transitions
    for i in range(5):
        dummy_s = np.random.rand(state_dim).astype(np.float32)
        dummy_a_road = np.random.randint(0, n_roads)
        dummy_a_upgrade = np.random.randint(0, n_upgrades)
        dummy_next_s = np.random.rand(state_dim).astype(np.float32) if i < 4 else None # Last one is terminal
        dummy_r = np.random.rand()
        dummy_d = True if i == 4 else False
        buffer.push(torch.from_numpy(dummy_s), torch.tensor([dummy_a_road]), torch.tensor([dummy_a_upgrade]), 
                    torch.from_numpy(dummy_next_s) if dummy_next_s is not None else None, 
                    torch.tensor([dummy_r]), torch.tensor([dummy_d]))
    
    print(f"Buffer length: {len(buffer)}")
    if len(buffer) >= 2:
        sample = buffer.sample(2)
        print("Sampled transitions:", sample)

