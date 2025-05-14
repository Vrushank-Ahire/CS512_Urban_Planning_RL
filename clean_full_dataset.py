import pandas as pd
import re
import numpy as np

# Function to parse vehicle volume by direction (adapted from previous script)
def parse_volume_by_direction(text):
    if pd.isna(text):
        return pd.Series([np.nan, np.nan, np.nan, np.nan], index=["Eastbound Volume", "Westbound Volume", "Northbound Volume", "Southbound Volume"])
    
    eb, wb, nb, sb = np.nan, np.nan, np.nan, np.nan
    
    eb_match = re.search(r"East Bound: (\d+)", str(text))
    if eb_match:
        eb = int(eb_match.group(1))
        
    wb_match = re.search(r"West Bound: (\d+)", str(text))
    if wb_match:
        wb = int(wb_match.group(1))
        
    nb_match = re.search(r"North Bound: (\d+)", str(text))
    if nb_match:
        nb = int(nb_match.group(1))
        
    sb_match = re.search(r"South Bound: (\d+)", str(text))
    if sb_match:
        sb = int(sb_match.group(1))
        
    return pd.Series([eb, wb, nb, sb], index=["Eastbound Volume", "Westbound Volume", "Northbound Volume", "Southbound Volume"])

# Load the new dataset
file_path = "/home/ubuntu/upload/average-daily-traffic-counts.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    exit()

print("--- Original Dataset Shape ---")
print(df.shape)

print("\n--- Original Dataset Head ---")
print(df.head())

print("\n--- Original Dataset Columns ---")
print(df.columns)

# Clean column names (strip leading/trailing spaces and replace spaces with underscores)
df.columns = df.columns.str.strip().str.replace(r"\s+-\s+", " - ", regex=True).str.replace(r"\s+", "_", regex=True)
# Specific rename for clarity if needed, e.g., ID_ to ID
if "ID_" in df.columns:
    df.rename(columns={"ID_": "ID"}, inplace=True)

print("\n--- Cleaned Dataset Columns ---")
print(df.columns)

# Parse 'Date_of_Count'
if "Date_of_Count" in df.columns:
    df["Date_of_Count"] = pd.to_datetime(df["Date_of_Count"], errors="coerce")

# Parse 'Vehicle_Volume_By_Each_Direction_of_Traffic'
if "Vehicle_Volume_By_Each_Direction_of_Traffic" in df.columns:
    directional_volumes = df["Vehicle_Volume_By_Each_Direction_of_Traffic"].apply(parse_volume_by_direction)
    df = pd.concat([df, directional_volumes], axis=1)
    # Fill NaNs in directional volumes with 0 after parsing, assuming no traffic if not specified
    for col in ["Eastbound_Volume", "Westbound_Volume", "Northbound_Volume", "Southbound_Volume"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
else:
    print("Warning: 'Vehicle_Volume_By_Each_Direction_of_Traffic' column not found.")
    # Create placeholder columns if the main one is missing, to prevent errors downstream
    df["Eastbound_Volume"] = 0
    df["Westbound_Volume"] = 0
    df["Northbound_Volume"] = 0
    df["Southbound_Volume"] = 0

# Select relevant columns (adjust based on actual needs and availability)
# Street Length is still noted as missing from the provided dataset columns
relevant_columns = [
    "ID", "Street", "Total_Passing_Vehicle_Volume", 
    "Latitude", "Longitude", "Zip_Codes", "Community_Areas", "Census_Tracts",
    "Eastbound_Volume", "Westbound_Volume", "Northbound_Volume", "Southbound_Volume"
]

# Filter for columns that actually exist in the dataframe to avoid KeyErrors
existing_relevant_columns = [col for col in relevant_columns if col in df.columns]
if len(existing_relevant_columns) < len(relevant_columns):
    missing_cols = set(relevant_columns) - set(existing_relevant_columns)
    print(f"\nWarning: The following relevant columns were not found and will be excluded: {missing_cols}")

df_cleaned = df[existing_relevant_columns].copy()

# Handle missing values in the selected columns (example: fill numerical with 0)
for col in df_cleaned.select_dtypes(include=np.number).columns:
    df_cleaned[col] = df_cleaned[col].fillna(0)
# For object columns, could fill with 'Unknown' or mode if necessary
# for col in df_cleaned.select_dtypes(include='object').columns:
#     df_cleaned[col] = df_cleaned[col].fillna('Unknown')

print("\n--- Missing Values in Cleaned Selected Data (after initial fillna(0) for numeric) ---")
print(df_cleaned.isnull().sum())

# Convert data types if necessary (example, ensure IDs are int, etc.)
if "ID" in df_cleaned.columns: df_cleaned["ID"] = df_cleaned["ID"].astype(int)
if "Total_Passing_Vehicle_Volume" in df_cleaned.columns: df_cleaned["Total_Passing_Vehicle_Volume"] = df_cleaned["Total_Passing_Vehicle_Volume"].astype(int)
if "Zip_Codes" in df_cleaned.columns: df_cleaned["Zip_Codes"] = df_cleaned["Zip_Codes"].astype(str) # Zip codes are often better as strings
if "Community_Areas" in df_cleaned.columns: df_cleaned["Community_Areas"] = df_cleaned["Community_Areas"].astype(str)
if "Census_Tracts" in df_cleaned.columns: df_cleaned["Census_Tracts"] = df_cleaned["Census_Tracts"].astype(str)

# Save the cleaned dataset
cleaned_file_path = "/home/ubuntu/cleaned_average_daily_traffic_counts.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"\n--- Cleaned Dataset Shape ({cleaned_file_path}) ---")
print(df_cleaned.shape)

print("\n--- Cleaned Dataset Head ---")
print(df_cleaned.head())

print("\n--- Cleaned Dataset Info ---")
df_cleaned.info()

print(f"\nData cleaning complete. Cleaned data saved to {cleaned_file_path}")

