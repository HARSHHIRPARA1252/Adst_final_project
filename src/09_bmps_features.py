import pandas as pd
import numpy as np

INPUT_FILE = "data/bmps_clean.csv"
OUTPUT_FILE = "data/bmps_features.csv"

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.sort_values(
    ["date", "match", "rank"]
).reset_index(drop=True)

df["win"] = (df["rank"] == 1).astype(int)

df["prev_avg_points"] = (
    df.groupby("team")["total_score"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

df["prev_avg_kills"] = (
    df.groupby("team")["elim_points"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

df["prev_win_rate"] = (
    df.groupby("team")["win"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

df["recent3_avg_points"] = (
    df.groupby("team")["total_score"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

df["recent3_avg_kills"] = (
    df.groupby("team")["elim_points"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

df["recent3_win_rate"] = (
    df.groupby("team")["win"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

df["prev_avg_points"] = df["prev_avg_points"].fillna(0)
df["prev_avg_kills"] = df["prev_avg_kills"].fillna(0)
df["prev_win_rate"] = df["prev_win_rate"].fillna(0)

df["recent3_avg_points"] = df["recent3_avg_points"].fillna(0)
df["recent3_avg_kills"] = df["recent3_avg_kills"].fillna(0)
df["recent3_win_rate"] = df["recent3_win_rate"].fillna(0)

match_previous_points = (
    df.groupby("match")["prev_avg_points"]
    .transform("mean")
)

match_previous_kills = (
    df.groupby("match")["prev_avg_kills"]
    .transform("mean")
)

df["relative_points"] = (
    df["prev_avg_points"] - match_previous_points
)

df["relative_kills"] = (
    df["prev_avg_kills"] - match_previous_kills
)

df["relative_points"] = df["relative_points"].fillna(0)
df["relative_kills"] = df["relative_kills"].fillna(0)

teams = df["team"].unique()

elo = {}

for team in teams:
    elo[team] = 1500.0

df["pre_match_elo"] = 0.0

df["elo_difference"] = 0.0

df["elo_probability"] = 0.5

for match_id in sorted(df["match"].unique()):

    match_data = df[df["match"] == match_id]

    match_teams = match_data["team"].tolist()

    match_mean_elo = np.mean(
        [elo[team] for team in match_teams]
    )

    for index in match_data.index:

        team = df.loc[index, "team"]

        current_elo = elo[team]

        df.loc[index, "pre_match_elo"] = current_elo

        difference = current_elo - match_mean_elo

        df.loc[index, "elo_difference"] = difference

        probability = (
            1 /
            (
                1 +
                10 ** (-difference / 400)
            )
        )

        df.loc[index, "elo_probability"] = probability

    for index in match_data.index:

        team = df.loc[index, "team"]

        rank = df.loc[index, "rank"]

        actual_score = 1 if rank == 1 else 0

        opponent_elos = [
            elo[other_team]
            for other_team in match_teams
            if other_team != team
        ]

        if len(opponent_elos) > 0:

            opponent_elo = np.mean(opponent_elos)

            expected_score = (
                1 /
                (
                    1 +
                    10 ** (
                        (opponent_elo - elo[team])
                        / 400
                    )
                )
            )

            elo[team] = (
                elo[team]
                +
                20 *
                (
                    actual_score -
                    expected_score
                )
            )

feature_columns = [
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

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("BMPS feature engineering completed.")
print("Shape:", df.shape)
print("Saved:", OUTPUT_FILE)

print("\nFeature columns:")

for feature in feature_columns:
    print("-", feature)

print("\nFirst 5 rows:")
print(
    df[
        [
            "match",
            "team",
            "rank",
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
    ].head()
)