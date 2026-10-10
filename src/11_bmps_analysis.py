import pandas as pd

df = pd.read_csv("data/bmps_predictions.csv")

for match in range(1, 19):
    temp = df[df["match"] == match]

    temp = temp[["team", "rank", "win_probability"]]
    temp = temp.sort_values("win_probability", ascending=False)

    print("\nMatch", match)
    print(temp.to_string(index=False))

    total = temp["win_probability"].sum()
    print("Probability total:", round(total, 2))