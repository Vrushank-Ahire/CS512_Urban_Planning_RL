import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from environment import UrbanPlanningEnv # Assuming environment.py is in the same directory
from agent import DQNAgent # Assuming agent.py is in the same directory

def generate_recommendations_and_visualize(env, agent, model_path, output_dir="/home/ubuntu/recommendations", n_episodes=3, max_t_per_episode=10):
    """Generates recommendations and visualizations from a trained DQN agent.

    Args:
        env (gym.Env): The environment instance.
        agent (DQNAgent): The DQN agent instance.
        model_path (str): Path to the saved model weights.
        output_dir (str): Directory to save recommendations and visualizations.
        n_episodes (int): Number of episodes to run for evaluation/recommendation generation.
        max_t_per_episode (int): Maximum timesteps per episode.
    """
    try:
        agent.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

    agent.policy_net.eval() # Set the network to evaluation mode

    all_episode_actions_summary = []
    recommendation_details = []

    print(f"\nGenerating recommendations for {n_episodes} simulated episodes...")

    os.makedirs(output_dir, exist_ok=True)

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_actions = []
        episode_summary = {
            "episode": i_episode,
            "initial_budget": env.current_budget,
            "initial_congestion": env._calculate_total_congestion_metric(),
            "actions": [],
            "final_budget": 0,
            "final_congestion": 0,
            "budget_spent": 0,
            "congestion_reduction": 0
        }

        print(f"\n--- Recommendation Episode: {i_episode} ---")
        print(f"Initial Budget: {episode_summary['initial_budget']}")
        print(f"Initial Total Congestion Metric: {episode_summary['initial_congestion']:.2f}")

        for t in range(max_t_per_episode):
            action_road, action_upgrade = agent.select_action(state, eps=0.0) # Greedy policy
            action = (action_road, action_upgrade)
            
            road_id = env.road_conditions.iloc[action_road]['id']
            upgrade_name = env.upgrade_types[action_upgrade]['name']
            upgrade_cost = env.upgrade_types[action_upgrade]['cost']
            
            action_detail = {
                "step": t + 1,
                "road_segment_idx": action_road,
                "road_id": road_id,
                "street_name": env.df_roads.iloc[action_road]['Street'],
                "upgrade_name": upgrade_name,
                "upgrade_cost": upgrade_cost,
                "budget_before_action": env.current_budget
            }
            print(f"  Step {t+1}: Recommend upgrading Road ID {road_id} (Street: {action_detail['street_name']}) with {upgrade_name}. Cost: {upgrade_cost}")

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            action_detail['reward_received'] = reward
            action_detail['budget_after_action'] = env.current_budget
            action_detail['congestion_after_action'] = env._calculate_total_congestion_metric()
            
            episode_actions.append(action_detail)
            recommendation_details.append(action_detail) # For overall summary
            
            state = next_state
            if done:
                break
        
        episode_summary['actions'] = episode_actions
        episode_summary['final_budget'] = env.current_budget
        episode_summary['final_congestion'] = env._calculate_total_congestion_metric()
        episode_summary['budget_spent'] = episode_summary['initial_budget'] - episode_summary['final_budget']
        episode_summary['congestion_reduction'] = episode_summary['initial_congestion'] - episode_summary['final_congestion']
        all_episode_actions_summary.append(episode_summary)

    recommendations_df = pd.DataFrame(recommendation_details)
    recommendations_file_path = os.path.join(output_dir, "investment_recommendations.csv")
    if not recommendations_df.empty:
        recommendations_df.to_csv(recommendations_file_path, index=False)
        print(f"\nInvestment recommendations saved to {recommendations_file_path}")

        plt.figure(figsize=(10, 6))
        recommendations_df['upgrade_name'].value_counts().plot(kind='bar')
        plt.title(f"Frequency of Recommended Upgrade Types (across {n_episodes} episodes)")
        plt.xlabel("Upgrade Type")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "upgrade_types_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Upgrade types distribution plot saved to {plot_path}")

        plt.figure(figsize=(12, 7))
        recommendations_df['street_name'].value_counts().nlargest(15).plot(kind='bar')
        plt.title(f"Top 15 Most Frequently Recommended Streets for Upgrade (across {n_episodes} episodes)")
        plt.xlabel("Street Name")
        plt.ylabel("Frequency of Recommendation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path_streets = os.path.join(output_dir, "top_streets_recommended.png")
        plt.savefig(plot_path_streets)
        plt.close()
        print(f"Top streets recommended plot saved to {plot_path_streets}")

        if "latitude" in env.df_roads.columns and "longitude" in env.df_roads.columns:
            merged_df = pd.merge(recommendations_df, env.df_roads[["id", "latitude", "longitude"]], left_on="road_id", right_on="id", how="left")
            merged_df.dropna(subset=["latitude", "longitude"], inplace=True)
            if not merged_df.empty:
                plt.figure(figsize=(10, 10))
                plt.scatter(merged_df["longitude"], merged_df["latitude"], alpha=0.6, c="red", label="Recommended Upgrades")
                plt.title(f"Geospatial Distribution of Recommended Upgrades (Sample from {n_episodes} episodes)")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.legend()
                plt.grid(True)
                map_plot_path = os.path.join(output_dir, "recommendations_map.png")
                plt.savefig(map_plot_path)
                plt.close()
                print(f"Recommendations map plot saved to {map_plot_path}")
            else:
                print("Skipping map plot: No valid coordinates found for recommended roads.")
        else:
            print("Skipping map plot: Latitude/Longitude data not available in road data.")
    else:
        print("No recommendations generated, skipping CSV and plot generation.")

    return recommendations_df, all_episode_actions_summary

if __name__ == "__main__":
    env = UrbanPlanningEnv(initial_budget=1000000, max_steps_per_episode=10)
    state_dim_from_env = env.observation_space.shape[0]
    num_roads_from_env = env.action_space.nvec[0]
    num_upgrades_from_env = env.action_space.nvec[1]

    agent = DQNAgent(state_size=state_dim_from_env, 
                     num_road_segments=num_roads_from_env, 
                     num_upgrade_types=num_upgrades_from_env)

    model_path = "/home/ubuntu/urban_planning_dqn_final.pth"
    
    recs_df, actions_summary = generate_recommendations_and_visualize(env, agent, model_path, n_episodes=3, max_t_per_episode=env.max_steps_per_episode)
    
    if recs_df is not None:
        print("\n--- Overall Recommendations Summary (First 5) ---")
        print(recs_df.head())
    
    # print("\n--- Detailed Actions per Episode ---")
    # for summary in actions_summary:
    #     print(f"Episode {summary['episode']}: Budget Spent: {summary['budget_spent']}, Congestion Reduced: {summary['congestion_reduction']:.2f}")
    #     for action in summary['actions']:
    #         print(f"  - {action['street_name']} ({action['road_id']}) upgraded with {action['upgrade_name']}")

    env.close()

