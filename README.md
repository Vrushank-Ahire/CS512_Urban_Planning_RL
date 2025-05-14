# Urban Planning RL Agent - FINAL_AI_CODE

This package contains the code and resources for an Urban Planning Reinforcement Learning (RL) agent designed to recommend road infrastructure upgrades.

## 1. Overview

The project uses a Deep Q-Network (DQN) agent to learn a policy for prioritizing road segment upgrades (e.g., widening, traffic signal installation, resurfacing) based on a dataset of traffic information. The goal is to simulate optimal infrastructure investment strategies to improve traffic flow and reduce congestion, considering budget constraints.

## 2. Folder Structure

```
FINAL_AI_CODE/
├── clean_full_dataset.py       # Script to clean the raw traffic data
├── environment.py              # Defines the RL environment (UrbanPlanningEnv)
├── model.py                    # Defines the DQN neural network architecture
├── agent.py                    # Defines the DQN agent logic
├── train.py                    # Script to train the DQN agent
├── evaluate_full_data.py       # Script to evaluate the trained agent
├── visualize.py                # Script to generate recommendations and visualizations
├── inspect_new_dataset.py      # Utility script for initial dataset inspection
├── documentation/              # Contains detailed reports and framework definitions
│   ├── urban_planning_rl_report_full_data.md
│   └── rl_framework_definition_full_data.md
├── models/                     # Contains trained model checkpoints (.pth files)
│   ├── urban_planning_dqn_episode_100.pth
│   ├── urban_planning_dqn_episode_200.pth
│   ├── urban_planning_dqn_episode_300.pth
│   ├── urban_planning_dqn_episode_400.pth
│   ├── urban_planning_dqn_episode_500.pth
│   └── urban_planning_dqn_final.pth
├── visualizations/             # Contains generated plots and charts
│   ├── training_scores.png
│   ├── upgrade_types_distribution.png
│   └── top_streets_recommended.png
└── README.md                   # This file
```

## 3. Setup and Dependencies

It is recommended to use a Python virtual environment.

**Python Version**: 3.11+

**Key Dependencies**:
*   `pandas`: For data manipulation.
*   `numpy`: For numerical operations.
*   `torch`: For the PyTorch deep learning framework.
*   `gymnasium`: For the RL environment framework (OpenAI Gym successor).
*   `matplotlib`: For plotting visualizations.

**Installation**:

1.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy torch torchvision torchaudio gymnasium matplotlib
    ```
    (Note: `torchvision` and `torchaudio` are often installed with `torch` but listed for completeness. Adjust based on your PyTorch installation needs, e.g., CPU-only or specific CUDA versions.)

## 4. Data

The primary dataset used for training and evaluation is `average-daily-traffic-counts.csv`. This file should be placed in a directory accessible by the scripts (e.g., an `upload` folder at the same level as `FINAL_AI_CODE`, or modify paths in scripts).

*   **Raw Data**: `average-daily-traffic-counts.csv` (user-provided, not included in this zip but required to run the cleaning script).
*   **Cleaned Data**: The `clean_full_dataset.py` script processes the raw CSV and outputs `cleaned_average_daily_traffic_counts.csv`. The environment (`environment.py`) expects this cleaned file by default in the `/home/ubuntu/` directory (as per the sandbox environment). You will need to adjust the path in `environment.py` (line 7) or place the cleaned file there.

## 5. Running the Code

**Important**: Ensure the `average-daily-traffic-counts.csv` file is available and paths are correctly set in the scripts if you are running outside the original sandbox environment.

1.  **Clean the Data**:
    *   Place `average-daily-traffic-counts.csv` in `/home/ubuntu/upload/` (or modify path in `clean_full_dataset.py`).
    *   Run the cleaning script:
        ```bash
        python clean_full_dataset.py
        ```
    *   This will generate `cleaned_average_daily_traffic_counts.csv` in `/home/ubuntu/`.

2.  **Train the Agent**:
    *   Ensure `cleaned_average_daily_traffic_counts.csv` is in `/home/ubuntu/` (or update path in `environment.py`).
    *   Run the training script:
        ```bash
        python train.py
        ```
    *   This will train the DQN agent and save model checkpoints (`.pth` files) in `/home/ubuntu/` (you might want to move them to the `FINAL_AI_CODE/models/` directory or adjust save paths in `train.py`). It also generates `training_scores.png`.

3.  **Evaluate the Agent**:
    *   Ensure the final trained model (`urban_planning_dqn_final.pth`) is in `/home/ubuntu/` (or update path in `evaluate_full_data.py`).
    *   Run the evaluation script:
        ```bash
        python evaluate_full_data.py
        ```
    *   This will print evaluation metrics to the console.

4.  **Generate Recommendations and Visualizations**:
    *   Ensure the final trained model is available as above.
    *   Run the visualization script:
        ```bash
        python visualize.py
        ```
    *   This will generate `investment_recommendations.csv`, `upgrade_types_distribution.png`, and `top_streets_recommended.png` in a `/home/ubuntu/recommendations/` directory (you might want to move them to `FINAL_AI_CODE/visualizations/` or adjust save paths).

## 6. Code Description

*   `inspect_new_dataset.py`: A utility script for initial pandas-based inspection of a new CSV dataset.
*   `clean_full_dataset.py`: Loads the raw `average-daily-traffic-counts.csv`, cleans column names, parses necessary fields (like `Date_of_Count`), attempts to parse directional traffic volumes (though this was noted as problematic with the full dataset_s `Vehicle_Volume_By_Each_Direction_of_Traffic` column format), selects relevant features, handles missing values, and saves the cleaned data to `cleaned_average_daily_traffic_counts.csv`.
*   `environment.py`: Implements the `UrbanPlanningEnv` class using `Gymnasium`. It loads the cleaned traffic data, defines the state and action spaces, and simulates the effects of infrastructure upgrades on a simplified congestion metric and budget.
*   `model.py`: Defines the `DQNetwork` class, a PyTorch neural network model used by the DQN agent. It features separate branches for predicting Q-values for road segment selection and upgrade type selection.
*   `agent.py`: Implements the `DQNAgent` class. This includes the DQN algorithm logic: action selection (epsilon-greedy), storing experiences in a replay buffer, learning from batches of experiences by updating the policy network, and soft-updating the target network.
*   `train.py`: The main script for training the `DQNAgent`. It initializes the environment and agent, runs the training loop for a specified number of episodes, saves model checkpoints, and plots the training scores.
*   `evaluate_full_data.py`: Loads a trained model and evaluates its performance over several episodes using a greedy policy (no exploration). It prints out actions taken and summary statistics.
*   `visualize.py`: Loads a trained model and runs it to generate investment recommendations. It saves these recommendations to a CSV file and creates visualizations: a bar chart of recommended upgrade types and a bar chart of the most frequently recommended streets. It also attempts to create a geospatial map if latitude/longitude data is correctly processed (this was noted as an issue in the last run).

## 7. Visualizations

The `visualizations/` folder contains:
*   `training_scores.png`: A plot showing the agent_s total reward per episode during training. This helps assess learning progress.
*   `upgrade_types_distribution.png`: A bar chart showing the frequency of different upgrade types recommended by the trained agent during the visualization run.
*   `top_streets_recommended.png`: A bar chart showing the most frequently recommended streets for upgrades by the trained agent.

(Note: The geospatial map `recommendations_map.png` was not generated in the last run due to an issue with accessing lat/lon data within the visualization script, even though the data is present in the cleaned CSV. This would require debugging in `visualize.py`.)

## 8. Documentation

The `documentation/` folder contains:
*   `urban_planning_rl_report_full_data.md`: The final detailed report on the project, methodology, findings, and future work, based on the full dataset.
*   `rl_framework_definition_full_data.md`: A document outlining the state, action, reward, and other components of the RL framework as applied to the full dataset.

## 9. Known Issues and Future Work

*   **Repetitive Policy**: The trained agent tends to learn a repetitive policy, often recommending the same road and upgrade type. This suggests a need for further refinement in state representation, reward shaping, exploration strategies, or simulation dynamics.
*   **Directional Volume Parsing**: The `clean_full_dataset.py` script noted issues with parsing individual directional volumes from the `Vehicle_Volume_By_Each_Direction_of_Traffic` column in the full dataset. If this data is crucial and available in a parsable format in the raw CSV, the parsing logic should be revisited.
*   **Street Length**: The "Street Length" feature, mentioned in the initial problem description, is not present in the provided `average-daily-traffic-counts.csv` and was thus excluded.
*   **Geospatial Map Visualization**: The `visualize.py` script failed to generate the `recommendations_map.png` due to an issue accessing latitude/longitude data for plotting, despite it being present in the cleaned data. This part of the script needs debugging.

Refer to the `urban_planning_rl_report_full_data.md` for a more detailed discussion of future work.

