import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

INPUT_FILE = "data/bmps_features.csv"
MODEL_FILE = "models/bmps_win_model.pkl"

df = pd.read_csv(INPUT_FILE)

df["target"] = (df["rank"] == 1).astype(int)

features = [
    "prev_avg_points",
    "prev_avg_kills",
    "prev_win_rate",
    "recent3_avg_points",
    "recent3_avg_kills",
    "recent3_win_rate",
    "relative_points",
    "relative_kills",
    "pre_match_elo",
    "elo_difference",
    "elo_probability"
]

X = df[features].fillna(0)
y = df["target"].fillna(0)

model = joblib.load(MODEL_FILE)

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)

print("BMPS MODEL EVALUATION")
print("=====================")

print("\nTotal rows:", len(X))
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(
    y,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nModel evaluation completed.")