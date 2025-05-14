import pandas as pd

# Load the new dataset
file_path = "/home/ubuntu/upload/average-daily-traffic-counts.csv"
df = pd.read_csv(file_path)

print("--- Dataset Head ---")
print(df.head())

print("\n--- Dataset Info ---")
df.info()

print("\n--- Dataset Description (Numerical Columns) ---")
print(df.describe())

print("\n--- Dataset Columns ---")
print(df.columns)

print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())

