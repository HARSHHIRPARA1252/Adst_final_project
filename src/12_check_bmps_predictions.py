import pandas as pd

df = pd.read_csv("data/bmps_predictions.csv")

for match in range(1, 19):
    temp = df[df["match"] == match]

    print("\nMatch:", match)

    print(
        temp[["team", "rank", "win_probability"]]
        .sort_values("win_probability", ascending=False)
        .to_string(index=False)
    )