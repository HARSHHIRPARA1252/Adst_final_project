import pandas as pd


# =========================================
# 1. LOAD FEATURE DATA
# =========================================

df = pd.read_csv("data/bgis_features.csv")

print("Rows:", len(df))


# =========================================
# 2. CREATE MATCH NUMBER
# =========================================

df["match_num"] = (
    df["match_id"]
    .astype(str)
    .str.replace("M", "", regex=False)
    .astype(int)
)


# =========================================
# 3. TEST MATCHES
# =========================================

test_matches = [13, 14, 15, 16, 17, 18]

test_df = df[
    df["match_num"].isin(test_matches)
].copy()


print("\nTesting matches:")
print(test_matches)

print("\nTesting rows:", len(test_df))


# =========================================
# 4. CREATE STRENGTH SCORE
# =========================================

test_df["strength_score"] = (

    test_df["prev_avg_points"] * 0.30

    + test_df["prev_avg_kills"] * 0.20

    + test_df["recent3_avg_points"] * 0.25

    + test_df["recent3_avg_kills"] * 0.15

    + test_df["prev_win_rate"] * 0.05

    + test_df["recent3_win_rate"] * 0.05
)


# =========================================
# 5. NORMALIZE SCORE INSIDE EACH MATCH
# =========================================

test_df["win_probability"] = (
    test_df
    .groupby("match_num")["strength_score"]
    .transform(
        lambda x: x / x.sum()
    )
)


# =========================================
# 6. RANK TEAMS
# =========================================

test_df["rank"] = (
    test_df
    .groupby("match_num")["strength_score"]
    .rank(
        ascending=False,
        method="first"
    )
)


# =========================================
# 7. SHOW ALL PREDICTIONS
# =========================================

print("\n=========================================")
print("BASELINE PREDICTIONS")
print("=========================================")


for match_num in test_matches:

    match_data = test_df[
        test_df["match_num"] == match_num
    ].sort_values(
        "strength_score",
        ascending=False
    )


    print("\nMatch:", f"M{match_num}")

    print(
        match_data[
            [
                "team",
                "strength_score",
                "win_probability",
                "rank",
                "win"
            ]
        ].to_string(index=False)
    )


# =========================================
# 8. TOP PREDICTED TEAM
# =========================================

top_predictions = (
    test_df
    .sort_values(
        ["match_num", "strength_score"],
        ascending=[True, False]
    )
    .groupby(
        "match_num",
        as_index=False
    )
    .first()
)


print("\n=========================================")
print("TOP PREDICTED TEAM PER MATCH")
print("=========================================")


print(
    top_predictions[
        [
            "team",
            "match_id",
            "strength_score",
            "win_probability",
            "win"
        ]
    ].to_string(index=False)
)


# =========================================
# 9. TOP-1 ACCURACY
# =========================================

top1_accuracy = (
    top_predictions["win"].mean()
)


print(
    "\nTop-1 accuracy:",
    round(top1_accuracy, 4)
)


# =========================================
# 10. ACTUAL WINNER RANK
# =========================================

print("\n=========================================")
print("ACTUAL WINNER RANK")
print("=========================================")


winner_ranks = []


for match_num in test_matches:

    match_data = test_df[
        test_df["match_num"] == match_num
    ].sort_values(
        "strength_score",
        ascending=False
    ).reset_index(drop=True)


    winner = match_data[
        match_data["win"] == 1
    ]


    if len(winner) == 1:

        winner_rank = int(
            winner.iloc[0].name
        ) + 1

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
# 11. AVERAGE WINNER RANK
# =========================================

if len(winner_ranks) > 0:

    average_rank = (
        sum(winner_ranks)
        / len(winner_ranks)
    )

    print(
        "\nAverage actual winner rank:",
        round(average_rank, 2)
    )


# =========================================
# 12. SAVE BASELINE RESULTS
# =========================================

test_df.to_csv(
    "outputs/bgis_baseline_predictions.csv",
    index=False
)


print(
    "\nBaseline predictions saved!"
)

print(
    "File: outputs/bgis_baseline_predictions.csv"
)