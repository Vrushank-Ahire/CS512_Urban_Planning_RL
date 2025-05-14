import gymnasium as gym
import numpy as np
import pandas as pd

class UrbanPlanningEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, data_path="/home/ubuntu/cleaned_average_daily_traffic_counts.csv", initial_budget=1000000, max_steps_per_episode=50):
        super(UrbanPlanningEnv, self).__init__()

        # Load and preprocess data
        try:
            self.df_roads = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: The data file {data_path} was not found. Please ensure clean_full_dataset.py has run successfully.")
            # Fallback to the sample data path if the main one is missing, for basic script integrity, but this is not ideal.
            # self.df_roads = pd.read_csv("/home/ubuntu/traffic_data_cleaned.csv") 
            raise # Re-raise the error to stop execution if the primary data file is missing
            
        self.num_road_segments = len(self.df_roads)
        if self.num_road_segments == 0:
            raise ValueError("Road dataset is empty. Cannot initialize environment.")

        # Define upgrade types, costs, and impacts (placeholder values)
        self.upgrade_types = {
            0: {"name": "Widening", "cost": 200000, "capacity_increase_factor": 0.20, "congestion_reduction_factor": 0.0}, # Example: Widening increases capacity
            1: {"name": "Traffic Signal", "cost": 50000, "congestion_reduction_factor": 0.10, "capacity_increase_factor": 0.0}, # Example: Signals reduce congestion directly
            2: {"name": "Resurfacing", "cost": 100000, "capacity_increase_factor": 0.05, "congestion_reduction_factor": 0.02} # Example: Resurfacing slightly improves capacity and reduces base congestion
        }
        self.num_upgrade_types = len(self.upgrade_types)

        self.action_space = gym.spaces.MultiDiscrete([self.num_road_segments, self.num_upgrade_types])

        # Simplified state: [budget_normalized, timestep_normalized, road1_congestion, ..., roadN_congestion]
        # Congestion values are expected to be between 0 and ~2 (can be higher)
        # Budget and timestep will be normalized for better learning.
        # Max budget is initial_budget, max_steps is max_steps_per_episode
        # For congestion, let's assume a practical max of 5 for normalization purposes, though it can exceed this.
        low_obs = np.array([0.0, 0.0] + [0.0] * self.num_road_segments) 
        high_obs = np.array([1.0, 1.0] + [5.0] * self.num_road_segments) # Normalized budget/step, max congestion for scaling
        self.observation_space = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.initial_budget = initial_budget
        self.max_steps_per_episode = max_steps_per_episode

        self.current_budget = 0
        self.current_step_in_episode = 0
        self.road_conditions = self._get_initial_road_conditions() # Initialize here to set up columns

        self.render_mode = "human"

        print(f"UrbanPlanningEnv initialized with {self.num_road_segments} road segments from {data_path}.")
        print(f"Action space: {self.action_space}")
        print(f"Observation space shape: {self.observation_space.shape}")

    def _get_initial_road_conditions(self):
        conditions = []
        for _, row in self.df_roads.iterrows():
            initial_capacity = row["Total_Passing_Vehicle_Volume"] * 1.5 
            if initial_capacity == 0: initial_capacity = 1.0 # Avoid division by zero
            
            congestion = row["Total_Passing_Vehicle_Volume"] / initial_capacity
            conditions.append({
                "id": row["ID"],
                "original_adt": row["Total_Passing_Vehicle_Volume"],
                "current_adt": row["Total_Passing_Vehicle_Volume"],
                "capacity": initial_capacity,
                "congestion": congestion,
                # "street_length": row.get("Street_Length_(miles)", 1.0), # Street Length is not in cleaned data, default to 1.0 for now if needed elsewhere
                "last_upgrade_step": -1,
                "upgrade_history": [] 
            })
        return pd.DataFrame(conditions)

    def _get_obs(self):
        # Normalized budget and timestep, then congestion levels
        norm_budget = self.current_budget / self.initial_budget if self.initial_budget > 0 else 0
        norm_timestep = self.current_step_in_episode / self.max_steps_per_episode if self.max_steps_per_episode > 0 else 0
        
        congestion_levels = self.road_conditions["congestion"].values
        # Clip congestion to the max defined in observation_space for stability, though ideally it shouldn_t exceed too much
        clipped_congestion = np.clip(congestion_levels, self.observation_space.low[2:], self.observation_space.high[2:])

        obs = np.concatenate(([norm_budget, norm_timestep], clipped_congestion)).astype(np.float32)
        return obs

    def _calculate_total_congestion_metric(self):
        # Sum of congestion levels. If street_length were available and reliable, (congestion * street_length) would be better.
        return self.road_conditions["congestion"].sum()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_budget = self.initial_budget
        self.current_step_in_episode = 0
        self.road_conditions = self._get_initial_road_conditions()
        return self._get_obs(), {}

    def step(self, action):
        self.current_step_in_episode += 1
        
        road_segment_idx, upgrade_type_idx = action
        # Ensure indices are within bounds
        if not (0 <= road_segment_idx < self.num_road_segments and 0 <= upgrade_type_idx < self.num_upgrade_types):
            # Invalid action, severe penalty, and end episode
            return self._get_obs(), -1000, True, False, {"error": "Invalid action indices"}
            
        selected_road_id = self.road_conditions.iloc[road_segment_idx]["id"]
        upgrade_details = self.upgrade_types[upgrade_type_idx]
        upgrade_cost = upgrade_details["cost"]

        terminated = False
        truncated = False
        reward = 0

        total_congestion_before = self._calculate_total_congestion_metric()

        if self.current_budget < upgrade_cost:
            reward = -100 
            terminated = True 
        else:
            self.current_budget -= upgrade_cost
            
            current_capacity = self.road_conditions.loc[road_segment_idx, "capacity"]
            current_congestion = self.road_conditions.loc[road_segment_idx, "congestion"]
            
            if upgrade_details.get("capacity_increase_factor", 0.0) > 0:
                current_capacity *= (1 + upgrade_details["capacity_increase_factor"])
                self.road_conditions.loc[road_segment_idx, "capacity"] = current_capacity
            
            # Apply direct congestion reduction factor
            if upgrade_details.get("congestion_reduction_factor", 0.0) > 0:
                 current_congestion *= (1 - upgrade_details["congestion_reduction_factor"])
            
            # Update congestion based on new capacity and potentially direct reduction
            current_adt = self.road_conditions.loc[road_segment_idx, "current_adt"]
            final_capacity = self.road_conditions.loc[road_segment_idx, "capacity"]
            if final_capacity == 0: final_capacity = 1.0
            
            # Recalculate congestion based on ADT/Capacity, then apply direct reduction factor if any
            base_congestion_after_capacity_change = current_adt / final_capacity
            final_congestion_on_road = base_congestion_after_capacity_change * (1 - upgrade_details.get("congestion_reduction_factor", 0.0))
            self.road_conditions.loc[road_segment_idx, "congestion"] = max(0, final_congestion_on_road) # Congestion cannot be negative

            self.road_conditions.loc[road_segment_idx, "last_upgrade_step"] = self.current_step_in_episode

            total_congestion_after = self._calculate_total_congestion_metric()
            congestion_reduction = total_congestion_before - total_congestion_after
            
            reward = (congestion_reduction * 100) - (upgrade_cost / 10000) # Adjusted reward scaling

        if self.current_step_in_episode >= self.max_steps_per_episode:
            truncated = True
        if self.current_budget <= 0:
            terminated = True
        
        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"--- Step: {self.current_step_in_episode} ---")
            print(f"Budget: {self.current_budget}")
            print(f"Total Congestion Metric (Sum): {self._calculate_total_congestion_metric():.2f}")
            print("---------------------")

    def close(self):
        pass

if __name__ == '__main__':
    try:
        env = UrbanPlanningEnv(max_steps_per_episode=10) # Use cleaned full dataset by default
        obs, info = env.reset()
        print("Initial Observation (Normalized Budget, Normalized Timestep, Congestions ...):")
        print(f"Shape: {obs.shape}, Min: {obs.min():.2f}, Max: {obs.max():.2f}, Mean: {obs.mean():.2f}")
        
        done = False
        total_reward = 0
        steps = 0
        for _ in range(2): # Run 2 sample episodes
            obs, info = env.reset()
            print(f"\nEpisode Start: Initial Budget: {env.current_budget}, Initial Congestion Sum: {env._calculate_total_congestion_metric():.2f}")
            episode_reward = 0
            for step_num in range(env.max_steps_per_episode + 5): # Allow a few more steps to see termination
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                print(f"  Step {step_num+1}: Action ({action[0]},{action[1]}), Reward: {reward:.2f}, Budget: {env.current_budget}, Term: {terminated}, Trunc: {truncated}")
                if done:
                    break
            print(f"Episode End: Total Reward: {episode_reward:.2f}, Final Budget: {env.current_budget}, Final Congestion Sum: {env._calculate_total_congestion_metric():.2f}")
        env.close()
    except Exception as e:
        print(f"Error during example usage: {e}")

