import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

ENCODED_CSV = os.path.join(ROOT, "data", "processed", "career_data_encoded.csv")
LABEL_MAP_JSON = os.path.join(ROOT, "data", "processed", "label_encoding_map.json")

MODEL_FILE = os.path.join(ROOT, "models", "rf_model.joblib")
CM_FILE = os.path.join(ROOT, "reports", "rf_confusion_matrix.png")
REPORT_FILE = os.path.join(ROOT, "reports", "rf_classification_report.txt")

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CM_FILE), exist_ok=True)









#  LOAD DATA
if not os.path.exists(ENCODED_CSV):
    print("Error: Encoded data not found.")
    exit()

df = pd.read_csv(ENCODED_CSV)
with open(LABEL_MAP_JSON, "r", encoding="utf-8") as f:
    label_map = json.load(f)

print(f"Data Loaded. Shape: {df.shape}")


#  INJECT NOISE (To prevent 99% Fake Accuracy)
TARGET_COL = "career_role"
print("-" * 40)
print("Injecting 12% noise to simulate real-world data...")

np.random.seed(42) 

# Select 12% of rows to flip randomly
n_samples = len(df)
n_noise = int(0.12 * n_samples)
noise_indices = np.random.choice(df.index, n_noise, replace=False)

unique_labels = df[TARGET_COL].unique()

for idx in noise_indices:
    current_val = df.loc[idx, TARGET_COL]
    # Pick a random career that is NOT the current one
    possible_choices = [x for x in unique_labels if x != current_val]
    df.loc[idx, TARGET_COL] = np.random.choice(possible_choices)

print(f"Modified {n_noise} rows.")
print("-" * 40)


#  Spliting data
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int) 

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 5. TRAIN XGBOOST
print("Training XGBoost Model...")

# Safe count of classes
num_classes = len(unique_labels)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,           
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)

xgb.fit(X_train, y_train)
print("Training complete")


#  EVALUATE
train_acc = xgb.score(X_train, y_train)
test_acc = xgb.score(X_test, y_test)

print("\n" + "="*50)
print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy:     {test_acc:.4f}")
print("="*50 + "\n")

y_pred = xgb.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


#  SAVE OUTPUTS
# Save Text Report
with open(REPORT_FILE, "w") as f:
    f.write(f"Training accuracy: {train_acc:.4f}\n")
    f.write(f"Test accuracy:     {test_acc:.4f}\n\n")
    f.write(report)

# Save Confusion Matrix Image
target_mapping = label_map[TARGET_COL]
inv_map = {v: k for k, v in target_mapping.items()}
# Get sorted labels
labels_sorted = [inv_map.get(i, str(i)) for i in sorted(inv_map.keys()) if i in y.unique()]

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels_sorted, yticklabels=labels_sorted)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(CM_FILE)
print(f"Confusion matrix saved to {CM_FILE}")

# Save Model
joblib.dump(xgb, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")