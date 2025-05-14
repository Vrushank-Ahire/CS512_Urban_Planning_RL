import torch
import numpy as np
import pandas as pd

from environment import UrbanPlanningEnv # Assuming environment.py is in the same directory
from agent import DQNAgent # Assuming agent.py is in the same directory

def evaluate_agent(env, agent, model_path, n_episodes=5, max_t_per_episode=50):
    """Evaluates a trained DQN agent.

    Args:
        env (gym.Env): The environment instance.
        agent (DQNAgent): The DQN agent instance.
        model_path (str): Path to the saved model weights.
        n_episodes (int): Number of episodes to run for evaluation.
        max_t_per_episode (int): Maximum timesteps per episode.
    """
    try:
        agent.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    agent.policy_net.eval() # Set the network to evaluation mode

    total_rewards_list = []
    final_budgets_list = []
    final_congestion_list = []

    print(f"\nStarting evaluation for {n_episodes} episodes...")

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        print(f"\n--- Evaluation Episode: {i_episode} ---")
        initial_budget_ep = env.current_budget
        initial_congestion_ep = env._calculate_total_congestion_metric()
        print(f"Initial Budget: {initial_budget_ep}")
        print(f"Initial Total Congestion Metric: {initial_congestion_ep:.2f}")

        for t in range(max_t_per_episode):
            action_road, action_upgrade = agent.select_action(state, eps=0.0) # Greedy policy for evaluation
            action = (action_road, action_upgrade)
            
            # Get details for logging before stepping
            road_id_log = env.road_conditions.iloc[action_road]["id"]
            street_name_log = env.df_roads.iloc[action_road]["Street"]
            upgrade_name_log = env.upgrade_types[action_upgrade]["name"]
            upgrade_cost_log = env.upgrade_types[action_upgrade]["cost"]
            budget_before_log = env.current_budget

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"  Step {t+1}: Action ({action_road},{action_upgrade}) -> Road ID {road_id_log} (\'{street_name_log}\'), Upgrade: {upgrade_name_log}, Cost: {upgrade_cost_log}")
            print(f"    Budget: {budget_before_log} -> {env.current_budget}. Reward: {reward:.2f}. Congestion: {env._calculate_total_congestion_metric():.2f}")
            
            state = next_state
            episode_reward += reward
            if done:
                break
        
        total_rewards_list.append(episode_reward)
        final_budgets_list.append(env.current_budget)
        final_congestion_list.append(env._calculate_total_congestion_metric())
        print(f"Episode {i_episode} finished after {t+1} steps. Total Reward: {episode_reward:.2f}")
        print(f"Final Budget: {env.current_budget}, Final Congestion: {env._calculate_total_congestion_metric():.2f}")

    print("\n--- Evaluation Summary ---")
    print(f"Average Total Reward over {n_episodes} episodes: {np.mean(total_rewards_list):.2f}")
    print(f"Average Final Budget over {n_episodes} episodes: {np.mean(final_budgets_list):.2f}")
    print(f"Average Final Congestion Metric over {n_episodes} episodes: {np.mean(final_congestion_list):.2f}")

if __name__ == "__main__":
    # Initialize environment - it will use the full cleaned dataset by default
    env = UrbanPlanningEnv(initial_budget=1000000, max_steps_per_episode=50) 

    state_dim_from_env = env.observation_space.shape[0]
    num_roads_from_env = env.action_space.nvec[0]
    num_upgrades_from_env = env.action_space.nvec[1]

    # Initialize agent
    agent = DQNAgent(state_size=state_dim_from_env, 
                     num_road_segments=num_roads_from_env, 
                     num_upgrade_types=num_upgrades_from_env)

    model_path = "/home/ubuntu/urban_planning_dqn_final.pth"
    
    evaluate_agent(env, agent, model_path, n_episodes=3, max_t_per_episode=env.max_steps_per_episode)
    
    env.close()

