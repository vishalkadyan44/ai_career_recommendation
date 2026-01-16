import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
# BASE_DIR -> /ai_learning/src/models

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  
# PROJECT_ROOT -> /ai_learning

INPUT_FILE = os.path.join(
    PROJECT_ROOT, "data", "processed", "career_data_encoded.csv"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT, "models", "logistic_regression_model.joblib"
)

PLOT_FILE = os.path.join(
    PROJECT_ROOT, "reports", "lr_confusion_matrix.png"
)

REPORT_FILE = os.path.join(
    PROJECT_ROOT, "reports", "lr_classification_report.txt"
)

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)



#  LOAD DATA
if not os.path.exists(INPUT_FILE):
    print("Error: Encoded data not found. Run feature_engineering.py first.")
    raise FileNotFoundError(f"File not found at {INPUT_FILE}. Please ensure feature engineering was successful.")

df = pd.read_csv(INPUT_FILE)
print(f"data loaded: {df.shape}")




#  Split data (remove leakage features)

leakage_cols = ["career_role", "gpa", "interestarea"]

X = df.drop(columns=leakage_cols)
y = df["career_role"]

print("Features used for training:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("data split done")

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("data split done")


#  Train model
print("training model...")

# max_iter=1000 helps prevent errors if the data is complex


# Train model
print("training model...")



try:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto"
    )
except TypeError:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )

model.fit(X_train, y_train)
print("training complete")



model = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])



model.fit(X_train, y_train)

print("training complete")




# Evaluate model
y_pred = model.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"accuracy: {acc:.4f}")

print("\nclassification report:")
print(classification_report(y_test, y_pred))


#  Visualisation (Confusion Matrix)
# This creates a picture showing where the model made mistakes
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)

# Get class labels for the confusion matrix for better readability
# Assuming `y` contains the original numeric labels after encoding
class_labels = np.unique(y)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(PLOT_FILE)
print("confusion matrix saved as png")


#  Save model
joblib.dump(model, MODEL_FILE)
print("model saved to joblib")
print("done")
