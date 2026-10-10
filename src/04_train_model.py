from itertools import combinations
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================================================
# BGIS MODEL V6
# ELO + RECENT FORM + LOGISTIC REGRESSION
# ============================================================

print("=" * 65)
print("BGIS WIN PREDICTION - MODEL V6")
print("ELO + RECENT FORM + LOGISTIC REGRESSION")
print("=" * 65)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/bgis_features.csv")

print("\nRows:", len(df))


# ============================================================
# 2. MATCH NUMBER
# ============================================================

df["match_num"] = (
    df["match_id"]
    .astype(str)
    .str.replace("M", "", regex=False)
    .astype(int)
)


# ============================================================
# 3. PRE-MATCH FEATURES
# ============================================================

base_features = [
    "prev_avg_points",
    "prev_avg_kills",
    "prev_win_rate",
    "recent3_avg_points",
    "recent3_avg_kills",
    "recent3_win_rate",
    "relative_points",
    "relative_kills",
    "relative_recent_points",
    "relative_recent_kills"
]


# ============================================================
# 4. CREATE ELO RATINGS
# ============================================================

print("\n" + "=" * 65)
print("CREATING ELO RATINGS")
print("=" * 65)


teams = sorted(
    df["team"].unique()
)


elo = {
    team: 1000.0
    for team in teams
}


K = 32


elo_history = []


for match_num in sorted(
    df["match_num"].unique()
):

    match = df[
        df["match_num"] == match_num
    ].copy()


    # Store rating BEFORE the match

    for _, row in match.iterrows():

        elo_history.append({

            "team": row["team"],

            "match_id": row["match_id"],

            "match_num": match_num,

            "pre_match_elo":
                elo[row["team"]]
        })


    # Find actual winner

    winners = match[
        match["win"] == 1
    ]


    if len(winners) != 1:

        continue


    winner = winners.iloc[0]["team"]


    # Average strength of opponents

    match_teams = list(
        match["team"]
    )


    old_ratings = {
        team: elo[team]
        for team in match_teams
    }


    average_elo = np.mean(
        list(old_ratings.values())
    )


    # Update each team

    for team in match_teams:

        current_elo = old_ratings[team]


        expected = (
            1 /
            (
                1 +
                10 ** (
                    (
                        average_elo
                        -
                        current_elo
                    ) / 400
                )
            )
        )


        actual = (
            1
            if team == winner
            else 0
        )


        elo[team] = (
            current_elo
            +
            K * (actual - expected)
        )


elo_df = pd.DataFrame(
    elo_history
)


print(
    "\nElo records:",
    len(elo_df)
)


# ============================================================
# 5. MERGE ELO
# ============================================================

df = df.merge(

    elo_df[
        [
            "team",
            "match_id",
            "match_num",
            "pre_match_elo"
        ]
    ],

    on=[
        "team",
        "match_id",
        "match_num"
    ],

    how="left"
)


# ============================================================
# 6. RELATIVE ELO
# ============================================================

match_mean_elo = (
    df.groupby("match_num")[
        "pre_match_elo"
    ]
    .transform("mean")
)


df["elo_difference"] = (
    df["pre_match_elo"]
    -
    match_mean_elo
)


# ============================================================
# 7. ELO PROBABILITY
# ============================================================

df["elo_probability"] = (

    1 /

    (
        1 +

        10 ** (

            (
                match_mean_elo
                -
                df["pre_match_elo"]
            )
            / 400
        )
    )
)


# ============================================================
# 8. REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "pre_match_elo"
    ]
).copy()


# ============================================================
# 9. TRAIN / TEST
# ============================================================

train_matches = list(
    range(2, 13)
)


test_matches = list(
    range(13, 19)
)


train_df = df[
    df["match_num"].isin(
        train_matches
    )
].copy()


test_df = df[
    df["match_num"].isin(
        test_matches
    )
].copy()


print("\n" + "=" * 65)
print("TRAIN / TEST SPLIT")
print("=" * 65)


print(
    "\nTrain matches:",
    train_matches
)


print(
    "Test matches:",
    test_matches
)


print(
    "\nTrain rows:",
    len(train_df)
)


print(
    "Test rows:",
    len(test_df)
)


# ============================================================
# 10. CREATE PAIRWISE DATA
#
# Target:
# 1 = Team A has stronger result
# 0 = Team B has stronger result
#
# IMPORTANT:
# Current match points/win are used ONLY for target creation.
# They are NOT model features.
# ============================================================

print("\n" + "=" * 65)
print("CREATING PAIRWISE TRAINING DATA")
print("=" * 65)


pair_rows = []


# Features available before the match

pair_base_features = [

    "prev_avg_points",
    "prev_avg_kills",
    "prev_win_rate",

    "recent3_avg_points",
    "recent3_avg_kills",
    "recent3_win_rate",

    "relative_points",
    "relative_kills",
    "relative_recent_points",
    "relative_recent_kills",

    "pre_match_elo",
    "elo_difference",
    "elo_probability"
]


for match_id, group in train_df.groupby(
    "match_id"
):

    group = group.reset_index(
        drop=True
    )


    for i in range(
        len(group)
    ):

        for j in range(
            i + 1,
            len(group)
        ):

            team_a = group.iloc[i]

            team_b = group.iloc[j]


            # Skip exact point ties

            if (
                team_a["points"]
                ==
                team_b["points"]
            ):

                continue


            target = (

                1

                if
                team_a["points"]
                >
                team_b["points"]

                else

                0
            )


            row = {

                "target":
                    target
            }


            for feature in pair_base_features:

                row[
                    feature + "_diff"
                ] = (

                    team_a[feature]
                    -
                    team_b[feature]
                )


            pair_rows.append(
                row
            )


pair_df = pd.DataFrame(
    pair_rows
)


print(
    "\nPairwise rows:",
    len(pair_df)
)


print(
    "\nTarget distribution:"
)


print(
    pair_df["target"]
    .value_counts()
)


# ============================================================
# 11. PAIRWISE FEATURES
# ============================================================

pair_features = [

    "prev_avg_points_diff",
    "prev_avg_kills_diff",
    "prev_win_rate_diff",

    "recent3_avg_points_diff",
    "recent3_avg_kills_diff",
    "recent3_win_rate_diff",

    "relative_points_diff",
    "relative_kills_diff",
    "relative_recent_points_diff",
    "relative_recent_kills_diff",

    "pre_match_elo_diff",
    "elo_difference_diff",
    "elo_probability_diff"
]


X_train = pair_df[
    pair_features
]


y_train = pair_df[
    "target"
]


# ============================================================
# 12. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 65)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 65)


model = Pipeline(

    [

        (
            "scaler",

            StandardScaler()
        ),

        (
            "classifier",

            LogisticRegression(

                C=1.0,

                max_iter=2000,

                class_weight="balanced",

                random_state=42
            )
        )
    ]
)


model.fit(
    X_train,
    y_train
)


print(
    "\nLogistic Regression trained successfully!"
)


# ============================================================
# 13. CREATE TEST PAIRS
# ============================================================

print("\n" + "=" * 65)
print("CREATING TEST PAIRS")
print("=" * 65)


test_pair_rows = []


for match_id, group in test_df.groupby(
    "match_id"
):

    group = group.reset_index(
        drop=True
    )


    for i in range(
        len(group)
    ):

        for j in range(
            i + 1,
            len(group)
        ):

            team_a = group.iloc[i]

            team_b = group.iloc[j]


            row = {

                "match_id":
                    match_id,

                "team_a":
                    team_a["team"],

                "team_b":
                    team_b["team"]
            }


            for feature in pair_base_features:

                row[
                    feature + "_diff"
                ] = (

                    team_a[feature]
                    -
                    team_b[feature]
                )


            test_pair_rows.append(
                row
            )


test_pairs = pd.DataFrame(
    test_pair_rows
)


print(
    "\nTest pairs:",
    len(test_pairs)
)


# ============================================================
# 14. PAIRWISE PREDICTION
# ============================================================

X_test = test_pairs[
    pair_features
]


test_pairs[
    "team_a_probability"
] = (

    model
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 15. PAIRWISE EVALUATION
# ============================================================

evaluation_rows = []


for _, row in test_pairs.iterrows():

    match_id = row["match_id"]


    match = test_df[
        test_df["match_id"]
        ==
        match_id
    ]


    winners = match[
        match["win"] == 1
    ]


    if len(winners) != 1:
        continue


    actual_winner = (
        winners.iloc[0]["team"]
    )


    actual = (

        1

        if row["team_a"]
        ==
        actual_winner

        else

        0
    )


    evaluation_rows.append({

        "actual":
            actual,

        "probability":
            row["team_a_probability"],

        "predicted":
            int(
                row["team_a_probability"]
                >= 0.5
            )
    })


evaluation_df = pd.DataFrame(
    evaluation_rows
)


pair_accuracy = accuracy_score(

    evaluation_df["actual"],

    evaluation_df["predicted"]
)


pair_auc = roc_auc_score(

    evaluation_df["actual"],

    evaluation_df["probability"]
)


print("\n" + "=" * 65)
print("PAIRWISE EVALUATION")
print("=" * 65)


print(
    f"\nPairwise Accuracy: "
    f"{pair_accuracy:.4f}"
)


print(
    f"Pairwise ROC-AUC: "
    f"{pair_auc:.4f}"
)


# ============================================================
# 16. RANK TEAMS
# ============================================================

print("\n" + "=" * 65)
print("RANKING TEAMS")
print("=" * 65)


ranking_rows = []


for match_id, group in test_df.groupby(
    "match_id"
):

    teams = list(
        group["team"]
    )


    scores = {
        team: 0.0
        for team in teams
    }


    # --------------------------------------------------------
    # Every team vs every other team
    # --------------------------------------------------------

    for team_a, team_b in combinations(
        teams,
        2
    ):

        pair = test_pairs[
            (
                test_pairs["match_id"]
                ==
                match_id
            )

            &

            (
                test_pairs["team_a"]
                ==
                team_a
            )

            &

            (
                test_pairs["team_b"]
                ==
                team_b
            )
        ]


        if len(pair) == 0:
            continue


        probability_a = float(
            pair.iloc[0][
                "team_a_probability"
            ]
        )


        probability_b = (
            1
            -
            probability_a
        )


        scores[
            team_a
        ] += probability_a


        scores[
            team_b
        ] += probability_b


    # Average score

    number_of_opponents = max(
        len(teams) - 1,
        1
    )


    for team in teams:

        scores[team] = (

            scores[team]

            /

            number_of_opponents
        )


    # --------------------------------------------------------
    # Convert ranking scores into probabilities
    # --------------------------------------------------------

    score_values = np.array(
        [
            scores[team]
            for team in teams
        ]
    )


    # Softmax

    temperature = 0.50


    scaled = (
        score_values
        /
        temperature
    )


    scaled = (
        scaled
        -
        np.max(scaled)
    )


    exp_values = np.exp(
        scaled
    )


    probabilities = (
        exp_values
        /
        exp_values.sum()
    )


    # Store

    for index, team in enumerate(
        teams
    ):

        actual_row = group[
            group["team"] == team
        ].iloc[0]


        ranking_rows.append({

            "match_id":
                match_id,

            "match_num":
                actual_row["match_num"],

            "team":
                team,

            "points":
                actual_row["points"],

            "win":
                actual_row["win"],

            "team_score":
                scores[team],

            "win_probability":
                probabilities[index]
        })


ranking_df = pd.DataFrame(
    ranking_rows
)


# ============================================================
# 17. RANK
# ============================================================

ranking_df[
    "predicted_rank"
] = (

    ranking_df

    .groupby("match_id")[
        "win_probability"
    ]

    .rank(
        ascending=False,
        method="first"
    )
)


# ============================================================
# 18. MATCH EVALUATION
# ============================================================

match_results = []


for match_id, group in ranking_df.groupby(
    "match_id"
):

    group = group.sort_values(
        "win_probability",
        ascending=False
    )


    predicted_winner = (
        group.iloc[0]["team"]
    )


    winners = group[
        group["win"] == 1
    ]


    if len(winners) != 1:
        continue


    actual_winner = (
        winners.iloc[0]["team"]
    )


    winner_rank = int(
        winners.iloc[0][
            "predicted_rank"
        ]
    )


    correct = (
        predicted_winner
        ==
        actual_winner
    )


    match_results.append({

        "match_id":
            match_id,

        "predicted_winner":
            predicted_winner,

        "actual_winner":
            actual_winner,

        "winner_rank":
            winner_rank,

        "correct":
            int(correct)
    })


match_results_df = pd.DataFrame(
    match_results
)


# ============================================================
# 19. TOP-1
# ============================================================

top1_accuracy = (

    match_results_df[
        "correct"
    ]
    .mean()
)


# ============================================================
# 20. TOP-3
# ============================================================

top3_correct = 0


for match_id, group in ranking_df.groupby(
    "match_id"
):

    group = group.sort_values(
        "win_probability",
        ascending=False
    )


    winners = group[
        group["win"] == 1
    ]


    if len(winners) != 1:
        continue


    actual_winner = (
        winners.iloc[0]["team"]
    )


    top3 = (
        group
        .head(3)["team"]
        .values
    )


    if actual_winner in top3:

        top3_correct += 1


total_matches = len(
    match_results_df
)


top3_accuracy = (

    top3_correct
    /
    total_matches
)


# ============================================================
# 21. AVERAGE WINNER RANK
# ============================================================

average_winner_rank = (

    match_results_df[
        "winner_rank"
    ]
    .mean()
)


# ============================================================
# 22. RESULTS
# ============================================================

print("\n" + "=" * 65)
print("BGIS MODEL V6 RESULTS")
print("=" * 65)


print(
    f"\nPairwise Accuracy: "
    f"{pair_accuracy:.4f}"
)


print(
    f"Pairwise ROC-AUC: "
    f"{pair_auc:.4f}"
)


print(
    f"Top-1 Match Accuracy: "
    f"{top1_accuracy:.4f}"
)


print(
    f"Top-3 Match Accuracy: "
    f"{top3_accuracy:.4f}"
)


print(
    f"Average Winner Rank: "
    f"{average_winner_rank:.2f}"
)


# ============================================================
# 23. SHOW PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("MATCH PREDICTIONS")
print("=" * 65)


for _, row in match_results_df.iterrows():

    print(
        f"\n{row['match_id']}"
    )


    print(
        "Predicted:",
        row["predicted_winner"]
    )


    print(
        "Actual:",
        row["actual_winner"]
    )


    print(
        "Actual winner rank:",
        row["winner_rank"]
    )


# ============================================================
# 24. FULL TOP PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("TOP PREDICTED TEAM PER MATCH")
print("=" * 65)


for match_id, group in ranking_df.groupby(
    "match_id"
):

    group = group.sort_values(
        "win_probability",
        ascending=False
    )


    top = group.iloc[0]


    actual = group[
        group["win"] == 1
    ].iloc[0]


    print(
        f"\n{match_id}"
    )


    print(
        "Predicted winner:",
        top["team"]
    )


    print(
        f"Probability: "
        f"{top['win_probability']:.2%}"
    )


    print(
        "Actual winner:",
        actual["team"]
    )


# ============================================================
# 25. SAVE PREDICTIONS
# ============================================================

ranking_df = ranking_df.sort_values(
    [
        "match_num",
        "predicted_rank"
    ]
)


ranking_df.to_csv(
    "outputs/bgis_v6_predictions.csv",
    index=False
)


print(
    "\nPredictions saved:"
)

print(
    "outputs/bgis_v6_predictions.csv"
)


# ============================================================
# 26. SAVE MATCH RESULTS
# ============================================================

match_results_df.to_csv(
    "outputs/bgis_v6_match_results.csv",
    index=False
)


print(
    "outputs/bgis_v6_match_results.csv"
)


# ============================================================
# 27. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/bgis_final_model_v6.pkl"
)


print(
    "Model saved:"
)

print(
    "models/bgis_final_model_v6.pkl"
)


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MODEL V6 COMPLETED")
print("=" * 65)

print(
    "\nTop-1:",
    round(
        top1_accuracy * 100,
        2
    ),
    "%"
)


print(
    "Top-3:",
    round(
        top3_accuracy * 100,
        2
    ),
    "%"
)


print(
    "Average Winner Rank:",
    round(
        average_winner_rank,
        2
    )
)