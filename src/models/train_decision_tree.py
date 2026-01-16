import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

INPUT_FILE = os.path.join(
    PROJECT_ROOT, "data", "processed", "career_data_encoded.csv"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT, "models", "decision_tree_model.joblib"
)

PLOT_FILE = os.path.join(
    PROJECT_ROOT, "reports", "dt_confusion_matrix.png"
)

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)

# Load data
if not os.path.exists(INPUT_FILE):
    print("Error: Encoded data not found. Run feature_engineering.py first.")
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)
print("Data loaded:", df.shape)

# Define features & target
leakage_cols = ["career_role", "gpa", "interestarea"]
X = df.drop(columns=leakage_cols)
y = df["career_role"]

print("Features used:", X.columns.tolist())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(
    random_state=42
)

print("Data split done")



model.fit(X_train, y_train)
print("Training complete")

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=labels,
    yticklabels=labels
)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(PLOT_FILE)
plt.close()

print("Confusion matrix saved")

# Save model
joblib.dump(model, MODEL_FILE)
print("Decision Tree model saved")
print("Done")
