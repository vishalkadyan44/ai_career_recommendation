import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "career_data_encoded.csv")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "svm_model.joblib")
PLOT_FILE = os.path.join(PROJECT_ROOT, "reports", "svm_confusion_matrix.png")

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)

# Load data
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("Run feature_engineering.py first")

df = pd.read_csv(INPUT_FILE)
print("Data loaded:", df.shape)

# Define features & target
leakage_cols = ["career_role", "gpa", "interestarea"]
X = df.drop(columns=leakage_cols)
y = df["career_role"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (Scaling required for SVM)
print("Training SVM...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=5,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)
print("Training complete")

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y)

sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(PLOT_FILE)
plt.close()

# Save model
joblib.dump(model, MODEL_FILE)
print("SVM model saved")
