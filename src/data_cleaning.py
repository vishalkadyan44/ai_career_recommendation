import pandas as pd
import numpy as np
import os


# current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root folder (ai_learning)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "career_data.csv")
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "career_data_cleaned.csv")



#  loading data 
print("Loading dataset...")

if not os.path.exists(RAW_DATA_PATH):
    print(f"Error: File not found at {RAW_DATA_PATH}")
    print("Please ensure 'career_data.csv' is in the same folder as this script.")
    exit()
else:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Original Data Shape: {df.shape}") 

#  Column name 
# Convert to lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

#  adding missing 'Gender' Column
# Simulating gender data since it is missing in the raw dataset
if 'gender' not in df.columns:
    np.random.seed(42)  # Ensures reproducibility
    df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
    print(" - 'gender' column created with random values.")

#  Remove Duplicates
df = df.drop_duplicates()

# Drop rows with any null values for data integrity
initial_rows = len(df)
df = df.dropna()
dropped_rows = initial_rows - len(df)
if dropped_rows > 0:
    print(f" - Removed {dropped_rows} rows containing missing values.")

# remove Outliers 
# filtering numeric columns to remove extreme values
print(" - Checking for outliers...")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # filter: Keep only data within valid bounds
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

#  Basic Text Cleanup
if 'skills' in df.columns:
    df['skills'] = df['skills'].str.strip()

#  saving the clean file 
print(f"Final Data Shape: {df.shape}")
df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"Success: Cleaned data saved to 'career_data_cleaned.csv'")

