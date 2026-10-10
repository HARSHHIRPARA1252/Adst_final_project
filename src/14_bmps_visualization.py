import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["feature"],
    importance["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("BMPS Model Feature Importance")

plt.tight_layout()

plt.savefig(
    "outputs/bmps_feature_importance.png",
    dpi=300
)

plt.show()

print("BMPS visualization completed.")
print("Saved: outputs/bmps_feature_importance.png")