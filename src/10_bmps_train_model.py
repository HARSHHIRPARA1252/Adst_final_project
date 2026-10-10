import pandas as pd
import joblib

df = pd.read_csv("data/bmps_features.csv")

model = joblib.load("models/bmps_win_model.pkl")

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

raw_probability = model.predict_proba(df[features])[:, 1]

df["raw_probability"] = raw_probability

df["win_probability"] = (
    df["raw_probability"]
    / df.groupby("match")["raw_probability"].transform("sum")
) * 100

df["win_probability"] = df["win_probability"].round(2)

df.to_csv("data/bmps_predictions.csv", index=False)

print("BMPS prediction completed.")
print("Saved: data/bmps_predictions.csv")

print("\nFirst 16 predictions:")

print(
    df[
        ["match", "team", "rank", "win_probability"]
    ].head(16).to_string(index=False)
)

print("\nMatch 1 probability total:")

print(
    df[df["match"] == 1]["win_probability"].sum()
)