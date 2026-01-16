import pandas as pd
import os
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "career_data_cleaned.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "career_data_encoded.csv")
MAPPING_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "label_encoding_map.json")


# Load data
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("Run data_cleaning.py first")

df = pd.read_csv(INPUT_FILE)
print("Data loaded:", df.shape)


# Create TARGET: career_role
def assign_career(row):
    interest = row["interestarea"].strip().lower()
    gpa = row["gpa"]

    if gpa >= 7.0:
        if interest == "computer science":
            return "Software Engineer"
        elif interest == "mathematics":
            return "Data Scientist"
        elif interest == "biology":
            return "Research Scientist"
        elif interest == "history":
            return "Policy Analyst"

    elif gpa >= 6.0:
        if interest == "computer science":
            return "IT Associate"
        elif interest == "mathematics":
            return "Data Analyst"
        elif interest == "biology":
            return "Lab Assistant"
        elif interest == "history":
            return "Content Analyst"

    else:
        return "General Management"

print("Generating target column: career_role")
df["career_role"] = df.apply(assign_career, axis=1)


# Separate target and features
target_col = "career_role"
X = df.drop(columns=[target_col])
y = df[target_col]


# Encode categorical FEATURES
label_maps = {}
le = LabelEncoder()

categorical_cols = X.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

    label_maps[col] = {
        str(cls): int(code)
        for cls, code in zip(le.classes_, le.transform(le.classes_))
    }
   

# Encode TARGET separately
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

label_maps[target_col] = {
    str(cls): int(code)
    for cls, code in zip(
        target_encoder.classes_,
        target_encoder.transform(target_encoder.classes_)
    )
}


# Scale numeric FEATURES only
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])

# Combine & save
df_final = X.copy()
df_final[target_col] = y_encoded

df_final.to_csv(OUTPUT_FILE, index=False)

with open(MAPPING_FILE, "w") as f:
    json.dump(label_maps, f, indent=4)

print("Feature engineering completed")
print("Target distribution:")
print(df_final[target_col].value_counts())
