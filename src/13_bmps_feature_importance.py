import pandas as pd
import joblib

INPUT_FILE = "data/bmps_features.csv"
MODEL_FILE = "models/bmps_win_model.pkl"

df = pd.read_csv(INPUT_FILE)

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

model = joblib.load(MODEL_FILE)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("BMPS FEATURE IMPORTANCE")
print("=======================")

print(importance.to_string(index=False))

print("\nMost important feature:")
print(importance.iloc[0]["feature"])

print("\nFeature importance analysis completed.")