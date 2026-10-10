import pandas as pd
import numpy as np
import joblib


# ============================================================
# BGIS FINAL PREDICTION ENGINE - V6
# ============================================================

print("=" * 60)
print("BGIS WIN PROBABILITY PREDICTOR")
print("Model: Elo + Recent Form + Logistic Regression")
print("=" * 60)


# ============================================================
# 1. LOAD FEATURE DATA
# ============================================================

df = pd.read_csv(
    "data/bgis_features.csv"
)


# ============================================================
# 2. CONVERT MATCH NUMBER
# ============================================================

df["match_num"] = (
    df["match_id"]
    .astype(str)
    .str.replace(
        "M",
        "",
        regex=False
    )
    .astype(int)
)


# ============================================================
# 3. LOAD FINAL V6 MODEL
# ============================================================

model = joblib.load(
    "models/bgis_final_model_v6.pkl"
)


print(
    "\nV6 model loaded successfully!"
)


# ============================================================
# 4. CREATE ELO RATINGS
# ============================================================

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


    # Store ratings before match

    for _, row in match.iterrows():

        elo_history.append({

            "team":
                row["team"],

            "match_id":
                row["match_id"],

            "match_num":
                match_num,

            "pre_match_elo":
                elo[row["team"]]
        })


    # Find winner

    winners = match[
        match["win"] == 1
    ]


    if len(winners) != 1:
        continue


    winner = winners.iloc[0]["team"]


    match_teams = list(
        match["team"]
    )


    old_ratings = {
        team: elo[team]
        for team in match_teams
    }


    average_elo = np.mean(
        list(
            old_ratings.values()
        )
    )


    # Update ratings

    for team in match_teams:

        current_elo = (
            old_ratings[team]
        )


        expected = (

            1 /

            (
                1 +

                10 ** (
                    (
                        average_elo
                        -
                        current_elo
                    )
                    / 400
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
            K * (
                actual
                -
                expected
            )
        )


elo_df = pd.DataFrame(
    elo_history
)


# ============================================================
# 5. MERGE ELO DATA
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
# 6. CALCULATE ELO FEATURES
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
# 7. GET LATEST AVAILABLE TEAM DATA
# ============================================================

df = df.sort_values(
    [
        "team",
        "match_num"
    ]
)


latest = (
    df
    .groupby("team")
    .tail(1)
    .copy()
)


# ============================================================
# 8. CLEAN TEAM NAMES
# ============================================================

latest["team_clean"] = (
    latest["team"]
    .astype(str)
    .str.strip()
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
)


latest["team_key"] = (
    latest["team_clean"]
    .str.casefold()
)


# ============================================================
# 9. SHOW TEAMS
# ============================================================

print(
    "\nAvailable teams:"
)

for team in latest[
    "team_clean"
]:

    print(
        "-",
        team
    )


# ============================================================
# 10. USER INPUT
# ============================================================

team_a_input = input(
    "\nEnter Team A: "
).strip()


team_b_input = input(
    "Enter Team B: "
).strip()


team_a_key = (
    " ".join(
        team_a_input.split()
    ).casefold()
)


team_b_key = (
    " ".join(
        team_b_input.split()
    ).casefold()
)


# ============================================================
# 11. CHECK TEAM A
# ============================================================

team_a_match = latest[
    latest["team_key"]
    ==
    team_a_key
]


if len(team_a_match) == 0:

    print(
        "\nTeam A not found."
    )

    print(
        "Please use a team from the list."
    )

    raise SystemExit


# ============================================================
# 12. CHECK TEAM B
# ============================================================

team_b_match = latest[
    latest["team_key"]
    ==
    team_b_key
]


if len(team_b_match) == 0:

    print(
        "\nTeam B not found."
    )

    print(
        "Please use a team from the list."
    )

    raise SystemExit


# ============================================================
# 13. SAME TEAM CHECK
# ============================================================

if team_a_key == team_b_key:

    print(
        "\nPlease select two different teams."
    )

    raise SystemExit


# ============================================================
# 14. GET TEAM DATA
# ============================================================

data_a = (
    team_a_match.iloc[0]
)


data_b = (
    team_b_match.iloc[0]
)


team_a = (
    data_a["team_clean"]
)


team_b = (
    data_b["team_clean"]
)


# ============================================================
# 15. CREATE PAIRWISE FEATURES
# ============================================================

features = [

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


pair_features = {}


for feature in features:

    pair_features[
        feature + "_diff"
    ] = (

        float(data_a[feature])
        -
        float(data_b[feature])
    )


# ============================================================
# 16. CREATE MODEL INPUT
# ============================================================

model_features = [

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


X_prediction = pd.DataFrame(
    [
        [
            pair_features[
                feature
            ]
            for feature in model_features
        ]
    ],
    columns=model_features
)


# ============================================================
# 17. PREDICT TEAM A PROBABILITY
# ============================================================

prob_a = float(
    model
    .predict_proba(
        X_prediction
    )[0][1]
)


# Team B is complementary probability

prob_b = (
    1.0
    -
    prob_a
)


# ============================================================
# 18. PREDICT WINNER
# ============================================================

if prob_a >= prob_b:

    predicted_winner = team_a

else:

    predicted_winner = team_b


# ============================================================
# 19. DISPLAY RESULT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MATCH WIN PROBABILITY"
)

print(
    "=" * 60
)


print(
    "\nTeam A:",
    team_a
)


print(
    "Win probability:",
    f"{prob_a * 100:.2f}%"
)


print(
    "\nTeam B:",
    team_b
)


print(
    "Win probability:",
    f"{prob_b * 100:.2f}%"
)


print(
    "\n----------------------------------------"
)


print(
    "Predicted winner:",
    predicted_winner
)


print(
    "----------------------------------------"
)


# ============================================================
# 20. TEAM INFORMATION
# ============================================================

print(
    "\nTEAM INFORMATION"
)


print(
    "\n",
    team_a
)


print(
    "Elo rating:",
    round(
        float(
            data_a["pre_match_elo"]
        ),
        2
    )
)


print(
    "Previous average points:",
    round(
        float(
            data_a[
                "prev_avg_points"
            ]
        ),
        2
    )
)


print(
    "Previous average kills:",
    round(
        float(
            data_a[
                "prev_avg_kills"
            ]
        ),
        2
    )
)


print(
    "Recent 3 match points:",
    round(
        float(
            data_a[
                "recent3_avg_points"
            ]
        ),
        2
    )
)


print(
    "\n",
    team_b
)


print(
    "Elo rating:",
    round(
        float(
            data_b["pre_match_elo"]
        ),
        2
    )
)


print(
    "Previous average points:",
    round(
        float(
            data_b[
                "prev_avg_points"
            ]
        ),
        2
    )
)


print(
    "Previous average kills:",
    round(
        float(
            data_b[
                "prev_avg_kills"
            ]
        ),
        2
    )
)


print(
    "Recent 3 match points:",
    round(
        float(
            data_b[
                "recent3_avg_points"
            ]
        ),
        2
    )
)


# ============================================================
# 21. PROBABILITY VALIDATION
# ============================================================

probability_total = (
    prob_a
    +
    prob_b
)


print(
    "\nProbability total:",
    f"{probability_total * 100:.2f}%"
)


if abs(
    probability_total - 1.0
) < 0.0001:

    print(
        "Probability check: PASS"
    )

else:

    print(
        "Probability check: FAILED"
    )


print(
    "\nPrediction completed successfully!"
)