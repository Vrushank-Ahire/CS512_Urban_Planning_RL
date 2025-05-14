# Reinforcement Learning Framework Definition (Full Dataset)

This document outlines the Reinforcement Learning (RL) framework for the Urban Planning Simulation project, updated to use the full `average-daily-traffic-counts.csv` dataset.

## 1. Goal

The primary goal is to use RL to recommend which road segments should be prioritized for upgrades (e.g., widening, traffic signal installation, resurfacing) to simulate improvements in overall traffic flow or reduction in average congestion, considering budget constraints.

## 2. Environment Components

### 2.1. State

The state representation for the RL agent will be a vector derived from the features available in the cleaned `cleaned_average_daily_traffic_counts.csv` dataset. For each road segment, the following features will be considered if available and relevant:

*   **ID**: Unique identifier for the road segment.
*   **Street**: Name of the street (can be used for context or potentially encoded).
*   **Total_Passing_Vehicle_Volume**: Represents the Average Daily Traffic (ADT) volume. This is a key indicator of road usage and potential congestion.
*   **Latitude**: Geographical latitude of the road segment. Useful for spatial analysis and visualization.
*   **Longitude**: Geographical longitude of the road segment. Useful for spatial analysis and visualization.
*   **Zip_Codes**: ZIP code where the road segment is located. Can be used as a categorical feature.
*   **Community_Areas**: Community area identifier. Can be used as a categorical feature.
*   **Census_Tracts**: Census tract identifier. Can be used as a categorical feature.

**Missing/Excluded Features from original request for this iteration:**
*   **Street Length**: This feature was mentioned in the initial problem description but is not present in the provided `average-daily-traffic-counts.csv` dataset. It will be excluded from the state unless it can be sourced externally.
*   **Directional Split (Eastbound, Westbound, Northbound, Southbound Volumes)**: The cleaning script for the new dataset indicated these columns were not found after parsing the `Vehicle_Volume_By_Each_Direction_of_Traffic` column. If this column is indeed missing or unparsable in the full dataset, these features cannot be included. The previous attempt with the sample data did parse these, so this needs to be re-verified with the actual full dataset structure during environment implementation. For now, we assume they might not be reliably available from the current cleaning output.

The complete state vector for the agent will likely be a flattened representation of these features for all road segments, plus the current remaining budget.

### 2.2. Action

The action space will be discrete and composite, consisting of two parts:

1.  **Road Segment Selection**: The agent chooses a specific road segment (by its index or ID) from the available segments in the dataset to apply an upgrade.
2.  **Upgrade Type Selection**: The agent chooses a type of upgrade to implement on the selected road segment. Examples of upgrade types include:
    *   Resurfacing (e.g., cost: $100,000, impact: moderate congestion reduction)
    *   Traffic Signal Installation/Optimization (e.g., cost: $250,000, impact: significant local congestion reduction, potential network effects)
    *   Lane Widening (e.g., cost: $1,000,000, impact: major capacity increase, significant congestion reduction)

The specific costs and impact factors for each upgrade type will need to be defined, potentially based on domain knowledge or estimations.

### 2.3. Reward

The reward function is crucial for guiding the agent_s learning process. It should incentivize actions that lead to desirable outcomes, such as reduced traffic congestion and efficient use of the budget.

A possible reward structure:

*   **Reward for Congestion Reduction**: `R_congestion = (Initial_Congestion_Metric - New_Congestion_Metric) * w1`
    *   Where `Initial_Congestion_Metric` and `New_Congestion_Metric` are measures of overall traffic congestion before and after the upgrade. This metric could be based on total ADT on highly utilized roads, simulated delays, or other relevant indicators.
    *   `w1` is a weighting factor.
*   **Penalty for Cost of Upgrade**: `R_cost = -Upgrade_Cost * w2`
    *   `Upgrade_Cost` is the cost of the implemented upgrade.
    *   `w2` is a weighting factor to balance cost against benefits.
*   **Penalty for Exceeding Budget**: A large negative reward if an action causes the budget to be exceeded, or if the agent attempts an action when the budget is insufficient. This encourages fiscal responsibility.
*   **Small Penalty per Step**: A small negative reward for each step taken could encourage the agent to find solutions more efficiently, though this needs careful tuning to avoid overly conservative behavior.

The total reward for a step would be `R_total = R_congestion + R_cost` (plus any other penalties/bonuses).

The definition of the "Congestion Metric" and the impact of upgrades on this metric will be a critical part of the environment simulation. Initially, this might be a simplified model (e.g., assuming a certain percentage reduction in a road_s contribution to overall congestion based on ADT and upgrade type).

## 3. RL Algorithm

A Deep Q-Network (DQN) or a similar value-based or policy-based algorithm suitable for discrete action spaces will be used. The choice will depend on the complexity of the state-action space and the desired performance.

## 4. Evaluation Metrics

The performance of the RL agent will be evaluated based on:

*   **Total Reward Accumulated**: Average total reward per episode during evaluation.
*   **Congestion Reduction Achieved**: The overall reduction in the defined congestion metric for a given budget.
*   **Budget Utilization**: How effectively the agent uses the allocated budget.
*   **Sensibility of Chosen Actions**: Qualitative analysis of the types of roads and upgrades prioritized by the agent.
*   **Diversity of Actions**: Whether the agent explores and implements a variety of sensible upgrade strategies.

This framework will guide the re-implementation of the RL environment and agent using the full dataset.
