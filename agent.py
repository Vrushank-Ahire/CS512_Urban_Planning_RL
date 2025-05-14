import torch
import torch.nn.functional as F # Added this import
import torch.optim as optim
import random
import numpy as np
from model import DQNetwork, ReplayBuffer, Transition # Assuming model.py is in the same directory or accessible

class DQNAgent:
    def __init__(self, state_size, num_road_segments, num_upgrade_types, 
                 replay_buffer_size=10000, batch_size=64, gamma=0.99, 
                 lr=1e-4, tau=1e-3, update_every=4, device=None):
        """
        Initializes a DQNAgent object.

        Args:
            state_size (int): Dimension of each state.
            num_road_segments (int): Number of road segments (for action_road branch).
            num_upgrade_types (int): Number of upgrade types (for action_upgrade branch).
            replay_buffer_size (int): Maximum size of replay buffer.
            batch_size (int): Minibatch size for training.
            gamma (float): Discount factor.
            lr (float): Learning rate for the optimizer.
            tau (float): For soft update of target parameters.
            update_every (int): How often to update the network.
            device (torch.device): Device to run the computations on (e.g., "cpu" or "cuda").
        """
        self.state_size = state_size
        self.num_road_segments = num_road_segments
        self.num_upgrade_types = num_upgrade_types
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_every = update_every

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # Q-Network
        self.policy_net = DQNetwork(state_size, num_road_segments, num_upgrade_types).to(self.device)
        self.target_net = DQNetwork(state_size, num_road_segments, num_upgrade_types).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Initialize target_net with policy_net weights
        self.target_net.eval() # Target network is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(replay_buffer_size)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action_road, action_upgrade, reward, next_state, done):
        # Save experience in replay memory
        self.memory.push(state, action_road, action_upgrade, next_state, reward, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def select_action(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_net.eval() # Set network to evaluation mode for action selection
        with torch.no_grad():
            q_roads, q_upgrades = self.policy_net(state_tensor)
        self.policy_net.train() # Set network back to train mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            # Greedy action
            action_road = q_roads.argmax(dim=1).item()
            action_upgrade = q_upgrades.argmax(dim=1).item()
        else:
            # Random action
            action_road = random.choice(list(range(self.num_road_segments)))
            action_upgrade = random.choice(list(range(self.num_upgrade_types)))
            
        return action_road, action_upgrade

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, ar, au, s
ze, d) tuples 
        """
        states, actions_road, actions_upgrade, next_states, rewards, dones = zip(*experiences)

        # Convert to PyTorch tensors
        states = torch.cat([s.unsqueeze(0) if s.dim() == 1 else s for s in states if s is not None]).float().to(self.device)
        
        # Handle cases where next_state might be None (terminal state)
        non_final_mask = torch.tensor([s is not None for s in next_states], device=self.device, dtype=torch.bool)
        non_final_next_states_list = [s for s in next_states if s is not None]
        if len(non_final_next_states_list) > 0:
             non_final_next_states = torch.cat([s.unsqueeze(0) if s.dim() == 1 else s for s in non_final_next_states_list]).float().to(self.device)
        else:
            non_final_next_states = torch.empty(0, self.state_size, device=self.device).float()

        actions_road = torch.cat(actions_road).unsqueeze(1).to(self.device) # Ensure it's a column vector
        actions_upgrade = torch.cat(actions_upgrade).unsqueeze(1).to(self.device) # Ensure it's a column vector
        rewards = torch.cat(rewards).unsqueeze(1).to(self.device) # Ensure it's a column vector
        dones = torch.cat(dones).unsqueeze(1).to(self.device)   # Ensure it's a column vector

        # Get Q values from policy net for current states and chosen actions
        # Q_policy(s_t, a_t) - we need Q values for the actions taken
        current_q_roads, current_q_upgrades = self.policy_net(states)
        q_expected_road = current_q_roads.gather(1, actions_road.long())
        q_expected_upgrade = current_q_upgrades.gather(1, actions_upgrade.long())
        
        # For dual-output DQN, the total Q-value for an action (road, upgrade) could be Q_road + Q_upgrade or some other combination.
        # Let's assume for now we train them somewhat independently or sum them for a combined loss.
        # A common approach for factored action spaces is to sum the Q-values: Q(s, (a_road, a_upgrade)) = Q_road(s, a_road) + Q_upgrade(s, a_upgrade)
        # Or, if the problem structure suggests, one might be primary and the other conditional.
        # The original task description mentioned "dual output branches: Branch 1: Q-values for road segment selection, Branch 2: Q-values for upgrade type selection"
        # This implies we might want to treat them as separate Q-value estimates that are combined or used sequentially.
        # For simplicity in this initial implementation, let's calculate separate losses and sum them, or average them.
        # This part needs careful consideration based on the problem formulation.
        # Let's assume for now we want to maximize the sum of Q-values for the chosen actions.
        q_expected_combined = q_expected_road + q_expected_upgrade

        # Get Q values from target net for next states
        # Q_target(s_{t+1}, argmax_a Q_policy(s_{t+1}, a)) for non-final next states
        q_target_next_roads_vals = torch.zeros(self.batch_size, device=self.device)
        q_target_next_upgrades_vals = torch.zeros(self.batch_size, device=self.device)
        
        if non_final_next_states.size(0) > 0: # Check if there are any non-final next states
            # For Double DQN, we select actions using the policy_net and evaluate using the target_net
            with torch.no_grad():
                next_q_roads_policy, next_q_upgrades_policy = self.policy_net(non_final_next_states)
                best_next_actions_road = next_q_roads_policy.argmax(1).unsqueeze(1)
                best_next_actions_upgrade = next_q_upgrades_policy.argmax(1).unsqueeze(1)
                
                next_q_roads_target, next_q_upgrades_target = self.target_net(non_final_next_states)
                q_target_next_roads_vals[non_final_mask] = next_q_roads_target.gather(1, best_next_actions_road.long()).squeeze()
                q_target_next_upgrades_vals[non_final_mask] = next_q_upgrades_target.gather(1, best_next_actions_upgrade.long()).squeeze()

        q_target_next_combined = q_target_next_roads_vals + q_target_next_upgrades_vals
        
        # Compute the target Q value: R + gamma * Q_target_next (if not done)
        q_targets = rewards.squeeze() + (self.gamma * q_target_next_combined * (~dones.squeeze()))

        # Compute Huber loss (or MSE loss)
        loss = F.smooth_l1_loss(q_expected_combined.squeeze(), q_targets.detach()) # .detach() because targets should not propagate gradients

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1) # Gradient clipping
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.policy_net, self.target_net)
        return loss.item()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval() # Set to eval mode after loading
        self.target_net.eval()
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    # Example usage (for testing the DQNAgent)
    state_dim = 11 
    n_roads = 9
    n_upgrades = 3
    agent = DQNAgent(state_dim, n_roads, n_upgrades, batch_size=2) # Small batch for testing
    print("DQNAgent initialized.")

    # Create dummy experiences
    for i in range(5):
        dummy_s = np.random.rand(state_dim).astype(np.float32)
        dummy_a_road, dummy_a_upgrade = agent.select_action(dummy_s, eps=1.0) # Explore
        dummy_next_s = np.random.rand(state_dim).astype(np.float32) if i < 4 else None
        dummy_r = np.random.rand()
        dummy_d = True if i == 4 else False
        
        # Convert to tensors before pushing to buffer, as expected by ReplayBuffer
        s_tensor = torch.from_numpy(dummy_s)
        ar_tensor = torch.tensor([dummy_a_road])
        au_tensor = torch.tensor([dummy_a_upgrade])
        next_s_tensor = torch.from_numpy(dummy_next_s) if dummy_next_s is not None else None
        r_tensor = torch.tensor([dummy_r])
        d_tensor = torch.tensor([dummy_d])
        
        agent.step(s_tensor, ar_tensor, au_tensor, r_tensor, next_s_tensor, d_tensor)
        print(f"Step {i+1}: memory length {len(agent.memory)}")

    if len(agent.memory) >= agent.batch_size:
        print("Attempting to learn from experiences...")
        loss = agent.learn(agent.memory.sample(agent.batch_size))
        print(f"Learning complete. Loss: {loss}")
    else:
        print("Not enough experiences to learn.")

    agent.save_model("/home/ubuntu/test_agent_model.pth")
    agent.load_model("/home/ubuntu/test_agent_model.pth")

