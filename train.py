import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from environment import UrbanPlanningEnv # Assuming environment.py is in the same directory
from agent import DQNAgent # Assuming agent.py is in the same directory
# model.py is imported by agent.py

def train_dqn(env, agent, n_episodes=2000, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning training loop.
    
    Args:
        env (gym.Env): The environment instance.
        agent (DQNAgent): The DQN agent instance.
        n_episodes (int): Maximum number of training episodes.
        max_t (int): Maximum number of timesteps per episode.
        eps_start (float): Starting value of epsilon, for epsilon-greedy action selection.
        eps_end (float): Minimum value of epsilon.
        eps_decay (float): Multiplicative factor (per episode) for decaying epsilon.
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    print(f"Starting training for {n_episodes} episodes...")
    
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_score = 0
        losses = []
        for t in range(max_t):
            action_road, action_upgrade = agent.select_action(state, eps)
            action = (action_road, action_upgrade) # Combine into a single action tuple for the env
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Convert to tensors before passing to agent.step
            s_tensor = torch.from_numpy(state).float().to(agent.device)
            ar_tensor = torch.tensor([action_road], device=agent.device)
            au_tensor = torch.tensor([action_upgrade], device=agent.device)
            next_s_tensor = torch.from_numpy(next_state).float().to(agent.device) if next_state is not None else None
            r_tensor = torch.tensor([reward], dtype=torch.float32, device=agent.device)
            d_tensor = torch.tensor([done], dtype=torch.bool, device=agent.device)
            
            loss = agent.step(s_tensor, ar_tensor, au_tensor, r_tensor, next_s_tensor, d_tensor)
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            episode_score += reward
            if done:
                break
                
        scores_window.append(episode_score)
        scores.append(episode_score)
        eps = max(eps_end, eps_decay * eps) # decay epsilon
        
        avg_loss = np.mean(losses) if losses else 0
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.4f}\tLoss: {avg_loss:.4f}", end="")
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.4f}\tLoss: {avg_loss:.4f}")
            agent.save_model(f"/home/ubuntu/urban_planning_dqn_episode_{i_episode}.pth")
            
    agent.save_model("/home/ubuntu/urban_planning_dqn_final.pth")
    print("\nTraining complete.")
    return scores

if __name__ == "__main__":
    # Initialize environment
    # The environment uses "/home/ubuntu/traffic_data_cleaned.csv" by default
    env = UrbanPlanningEnv(initial_budget=1000000, max_steps_per_episode=50) # Use max_steps_per_episode from env

    # Get state and action sizes from the environment
    # state_size = env.observation_space.shape[0]
    # For MultiDiscrete action space, we need the number of options for each part
    # num_road_segments = env.action_space.nvec[0]
    # num_upgrade_types = env.action_space.nvec[1]
    
    # Hacky way to get state_size, num_road_segments, num_upgrade_types if not directly accessible
    # This should ideally be exposed by the environment more cleanly or derived from its properties
    # From environment.py: obs_space shape is (11,) for the sample data (9 roads + budget + timestep)
    # Action space: MultiDiscrete([num_road_segments, num_upgrade_types]) -> num_road_segments=9, num_upgrade_types=3
    # These were hardcoded in model.py and agent.py test sections, let's use them for consistency for now.
    # A better way is to get them from env.observation_space.shape[0] and env.action_space.nvec
    state_dim_from_env = env.observation_space.shape[0]
    num_roads_from_env = env.action_space.nvec[0]
    num_upgrades_from_env = env.action_space.nvec[1]

    print(f"State dimension from env: {state_dim_from_env}")
    print(f"Number of roads from env: {num_roads_from_env}")
    print(f"Number of upgrade types from env: {num_upgrades_from_env}")

    # Initialize agent
    agent = DQNAgent(state_size=state_dim_from_env, 
                     num_road_segments=num_roads_from_env, 
                     num_upgrade_types=num_upgrades_from_env,
                     replay_buffer_size=100000, # Increased buffer size
                     batch_size=128,          # Increased batch size
                     gamma=0.99,
                     lr=5e-5,               # Adjusted learning rate
                     tau=1e-3,
                     update_every=4)

    # Train the agent
    # Using fewer episodes for a quick test run. For real training, this would be much higher.
    scores = train_dqn(env, agent, n_episodes=500, max_t=env.max_steps_per_episode, eps_decay=0.99)

    # Plot scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.title("Training Scores")
    plt.savefig("/home/ubuntu/training_scores.png")
    print("Training scores plot saved to /home/ubuntu/training_scores.png")
    # plt.show() # Would require X11 forwarding if run in a headless environment

    env.close()

