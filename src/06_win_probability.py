import pandas as pd
import numpy as np


# =========================================
# 1. LOAD FEATURE DATA
# =========================================

df = pd.read_csv("data/bgis_features.csv")

print("Rows:", len(df))


# =========================================
# 2. CONVERT MATCH ID TO NUMBER
# =========================================

df["match_num"] = (
    df["match_id"]
    .astype(str)
    .str.replace("M", "", regex=False)
    .astype(int)
)


# =========================================
# 3. SELECT TEST MATCHES
# =========================================

test_matches = list(range(13, 19))

test_df = df[
    df["match_num"].isin(test_matches)
].copy()

print("Test rows:", len(test_df))


# =========================================
# 4. CALCULATE TEAM STRENGTH
# =========================================

test_df["strength_score"] = (

    test_df["prev_avg_points"] * 0.30

    + test_df["prev_avg_kills"] * 0.15

    + test_df["prev_win_rate"] * 0.20

    + test_df["recent3_avg_points"] * 0.20

    + test_df["recent3_avg_kills"] * 0.10

    + test_df["recent3_win_rate"] * 0.05
)


# =========================================
# 5. CALCULATE WIN PROBABILITY
# =========================================

test_df["win_probability"] = 0.0

temperature = 2.0


for match_num in test_matches:

    mask = (
        test_df["match_num"] == match_num
    )

    scores = test_df.loc[
        mask,
        "strength_score"
    ].values

    # Prevent numerical overflow
    adjusted_scores = (
        scores - scores.max()
    ) / temperature

    exp_scores = np.exp(
        adjusted_scores
    )

    probabilities = (
        exp_scores
        / exp_scores.sum()
    )

    test_df.loc[
        mask,
        "win_probability"
    ] = probabilities


# =========================================
# 6. PREDICTED RANK
# =========================================

test_df["predicted_rank"] = (
    test_df
    .groupby("match_num")[
        "strength_score"
    ]
    .rank(
        ascending=False,
        method="first"
    )
)


# =========================================
# 7. CHECK PROBABILITY SUM
# =========================================

print("\n=========================================")
print("PROBABILITY CHECK")
print("=========================================")

probability_check = (
    test_df
    .groupby("match_num")[
        "win_probability"
    ]
    .sum()
)

print(
    probability_check
)


# =========================================
# 8. TOP PREDICTED TEAM PER MATCH
# =========================================

top_predictions = (
    test_df
    .sort_values(
        [
            "match_num",
            "win_probability"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "match_num",
        as_index=False
    )
    .first()
)


# =========================================
# 9. DISPLAY WIN PROBABILITY
# =========================================

print("\n=========================================")
print("WIN PROBABILITY PREDICTIONS")
print("=========================================")

print(
    top_predictions[
        [
            "team",
            "match_id",
            "strength_score",
            "win_probability",
            "predicted_rank",
            "win"
        ]
    ].to_string(
        index=False
    )
)


# =========================================
# 10. TOP 5 TEAMS PER MATCH
# =========================================

print("\n=========================================")
print("TOP 5 TEAMS PER MATCH")
print("=========================================")


for match_num in test_matches:

    match_data = test_df[
        test_df["match_num"] == match_num
    ].sort_values(
        "win_probability",
        ascending=False
    ).head(5)


    print(
        f"\nMatch M{match_num}"
    )

    print(
        match_data[
            [
                "team",
                "win_probability",
                "predicted_rank",
                "win"
            ]
        ].to_string(
            index=False
        )
    )


# =========================================
# 11. TOP-1 ACCURACY
# =========================================

top1_accuracy = (
    top_predictions["win"].mean()
)


print("\n=========================================")
print("TOP-1 ACCURACY")
print("=========================================")

print(
    "Top-1 match accuracy:",
    round(
        top1_accuracy,
        4
    )
)


# =========================================
# 12. ACTUAL WINNER RANK
# =========================================

print("\n=========================================")
print("ACTUAL WINNER RANK")
print("=========================================")


winner_ranks = []


for match_num in test_matches:

    match_data = test_df[
        test_df["match_num"] == match_num
    ].sort_values(
        "win_probability",
        ascending=False
    ).reset_index(
        drop=True
    )


    winner = match_data[
        match_data["win"] == 1
    ]


    if len(winner) == 1:

        winner_rank = (
            winner.index[0] + 1
        )

        winner_ranks.append(
            winner_rank
        )

        print(
            f"M{match_num} winner:",
            winner.iloc[0]["team"],
            "| Rank:",
            winner_rank
        )


# =========================================
# 13. AVERAGE WINNER RANK
# =========================================

if len(winner_ranks) > 0:

    average_rank = (
        sum(winner_ranks)
        / len(winner_ranks)
    )

    print(
        "\nAverage actual winner rank:",
        round(
            average_rank,
            2
        )
    )


# =========================================
# 14. SAVE RESULTS
# =========================================

test_df.to_csv(
    "outputs/bgis_win_probability.csv",
    index=False
)


print(
    "\nWin probability results saved!"
)

print(
    "File: outputs/bgis_win_probability.csv"
)


# =========================================
# 15. FINISHED
# =========================================

print(
    "\nWin probability model completed successfully!"
)