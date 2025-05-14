# Urban Planning Simulation using Reinforcement Learning: Final Report (Full Dataset)

## 1. Introduction

This project aimed to develop a Reinforcement Learning (RL) agent to recommend road segments for upgrades to improve urban infrastructure, based on the provided `average-daily-traffic-counts.csv` dataset (1279 entries). The goal was to simulate optimal infrastructure investment strategies, considering factors like traffic volume and location data, to guide city planners.

## 2. Data Processing (Full Dataset)

The `average-daily-traffic-counts.csv` dataset was processed as follows:

*   **Data Loading and Initial Inspection**: The CSV file was loaded into a pandas DataFrame. Initial inspection revealed 1279 entries and 15 columns.
*   **Column Name Cleaning**: Column names were stripped of leading/trailing spaces and standardized (e.g., spaces replaced with underscores).
*   **Feature Parsing and Selection**:
    *   `Date_of_Count` was converted to datetime objects.
    *   The `Vehicle_Volume_By_Each_Direction_of_Traffic` column was processed to extract directional volumes. However, the cleaning script (`clean_full_dataset.py`) output indicated that after parsing this column from the full dataset, the individual directional volume columns (Eastbound_Volume, Westbound_Volume, etc.) were not found or populated as expected. This is a key difference from the initial sample data processing and means these specific directional splits were not available for the RL state in this iteration.
    *   Relevant columns selected for the cleaned dataset (`cleaned_average_daily_traffic_counts.csv`) included: `ID`, `Street`, `Total_Passing_Vehicle_Volume`, `Latitude`, `Longitude`, `Zip_Codes`, `Community_Areas`, and `Census_Tracts`.
*   **Missing Value Handling**: The dataset had no missing values in the initially loaded columns. Numerical features in the selected subset were filled with 0 if any NaNs were introduced during processing, and categorical identifiers like `Zip_Codes` were treated as strings.
*   **Street Length**: This feature, mentioned in the original problem description, was not present in the provided CSV and was therefore excluded.

## 3. Reinforcement Learning Framework (Full Dataset)

The RL problem was defined (`rl_framework_definition_full_data.md`) considering the full dataset:

*   **State**: A vector including normalized current budget, normalized current timestep, and the current congestion level for each of the 1279 road segments. Congestion was calculated as `current_ADT / capacity`. Features like `Latitude`, `Longitude`, `Zip_Codes`, `Community_Areas`, `Census_Tracts` from the cleaned data were available to the environment for context but not directly part of the flattened state vector fed to the DQN agent in this simplified implementation. A more complex state could explicitly encode these.
*   **Action**: A composite discrete action: (1) select one of the 1279 road segments, and (2) select one of three upgrade types (Widening, Traffic Signal, Resurfacing), each with defined costs and impact factors on capacity or congestion.
*   **Reward**: A function combining the reduction in the total congestion metric (sum of congestion on all roads) and the cost of the upgrade. Penalties were applied for attempting unaffordable upgrades or exceeding the budget.

## 4. RL Environment Implementation (Full Dataset)

The `UrbanPlanningEnv` class (`environment.py`) was updated:

*   It now loads `cleaned_average_daily_traffic_counts.csv` (1279 road segments).
*   The observation space was defined as a `Box` space with shape `(1281,)` (normalized budget, normalized timestep, 1279 congestion values).
*   The action space is `MultiDiscrete([1279, 3])`.
*   Initial road conditions (capacity, congestion) are derived from the `Total_Passing_Vehicle_Volume` in the loaded dataset.
*   The `step()` method applies upgrades, updates budget and road conditions, and calculates rewards.

## 5. RL Agent Implementation and Training (Full Dataset)

*   **Algorithm**: A Deep Q-Network (DQN) agent (`agent.py`, `model.py`) was used.
*   **Training Process** (`train.py`):
    *   The agent was trained for 500 episodes using the updated environment.
    *   Hyperparameters: learning rate 5e-5, gamma 0.99, epsilon decayed from 1.0 to 0.01, replay buffer size 100,000, batch size 128.
    *   The final model was saved as `urban_planning_dqn_final.pth`.
    *   Training scores (average reward per 100 episodes) were plotted (`training_scores.png`). The scores remained consistently negative, around -85 to -90, suggesting that the costs of upgrades generally outweighed the simulated congestion reduction benefits under the current reward structure and simulation model.

## 6. Evaluation and Validation (Full Dataset)

*   **Evaluation Process** (`evaluate_full_data.py`):
    *   The trained agent was evaluated for 3 episodes using a greedy policy.
    *   **Observed Behavior**: The agent consistently chose to upgrade the same road segment (Road ID 782, Street: Milwaukee Ave) with the same upgrade type ("Traffic Signal") in every step of every evaluation episode until the budget was depleted. This indicates a highly repetitive and likely suboptimal policy.
*   **Interpretation of Repetitive Policy**: This behavior, similar to what was seen with the sample data but now confirmed on the full dataset, suggests potential issues:
    *   **Reward Shaping**: The reward function might not be sufficiently nuanced to encourage diverse strategies. The agent might have found a local optimum where one specific action yields a marginally better (or less negative) immediate reward.
    *   **State Representation**: The simplified state (primarily congestion levels) might not provide enough information for the agent to make more complex decisions.
    *   **Exploration**: The exploration during training might not have been sufficient to discover more diverse, effective policies.
    *   **Simulation Dynamics**: The way upgrades impact congestion in the simulation might be too simplistic or favor certain actions disproportionately.

## 7. Recommendations and Visualizations (Full Dataset)

Recommendations were generated based on the agent_s learned policy (`visualize.py`):

*   **Investment Recommendations**: `recommendations/investment_recommendations.csv` lists the sequence of upgrades. Due to the agent_s policy, these are repetitive (Milwaukee Ave, Traffic Signal).
*   **Visualizations** (in `/home/ubuntu/recommendations/`):
    *   `upgrade_types_distribution.png`: Shows "Traffic Signal" as the overwhelmingly recommended upgrade.
    *   `top_streets_recommended.png`: Shows "Milwaukee Ave" as the overwhelmingly recommended street.
    *   **Geospatial Map**: The `visualize.py` script noted "Skipping map plot: Latitude/Longitude data not available in road data." This indicates an issue in how the `UrbanPlanningEnv` or the `visualize.py` script itself accesses the latitude/longitude columns from `env.df_roads` when merging for the plot, even though these columns *are* present in `cleaned_average_daily_traffic_counts.csv`. This is a bug in the visualization part of the workflow that needs to be addressed for map generation.

## 8. Conclusion and Future Work

This project successfully re-executed the RL pipeline using the full 1279-entry dataset. While the framework is functional, the trained DQN agent developed a highly repetitive policy, indicating that further refinements are necessary for practical urban planning recommendations.

**Key areas for Future Work remain consistent with the previous report, but with emphasis on the full dataset context**:

*   **Refine Reward Function and State Representation**: This is critical to encourage more diverse and effective policies.
*   **Improve Simulation Dynamics**: More realistic modeling of upgrade impacts.
*   **Advanced RL Algorithms & Hyperparameter Tuning**: Explore alternatives to DQN or more sophisticated DQN variants.
*   **Address Visualization Bug**: Ensure the geospatial map visualization correctly uses the available latitude/longitude data from the full dataset.
*   **Incorporate Missing Data**: If possible, source "Street Length" and investigate the parsing of "Vehicle Volume By Each Direction of Traffic" in the full CSV to include directional splits if the raw data supports it.

## 9. Deliverables

*   **This Report**: `urban_planning_rl_report_full_data.md`
*   **To-Do Checklist**: `todo.md` (updated for full dataset run)
*   **RL Framework Definition**: `rl_framework_definition_full_data.md`
*   **Cleaned Full Dataset**: `cleaned_average_daily_traffic_counts.csv`
*   **Python Scripts**: `clean_full_dataset.py`, `environment.py`, `model.py`, `agent.py`, `train.py`, `evaluate_full_data.py`, `visualize.py`
*   **RL Model Checkpoint (Full Data)**: `urban_planning_dqn_final.pth`
*   **Training Progress (Full Data)**: `training_scores.png`
*   **Recommendations Output (Full Data)** (in `recommendations` directory):
    *   `investment_recommendations.csv`
    *   `upgrade_types_distribution.png`
    *   `top_streets_recommended.png`


