import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

raw_df = pd.read_csv(
    "data/bgis.csv"
)

df = pd.read_csv(
    "data/bgis_features.csv"
)

print("Data loaded successfully!")


# ============================================================
# CONVERT MATCH ID
# ============================================================

raw_df["match_num"] = (
    raw_df["match_id"]
    .astype(str)
    .str.replace(
        "M",
        "",
        regex=False
    )
    .astype(int)
)


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
# LOAD MODEL
# ============================================================

print("\nLoading V6 model...")

model = joblib.load(
    "models/bgis_final_model_v6.pkl"
)

print("Model loaded successfully!")


# ============================================================
# GET MODEL FEATURES
# ============================================================

if hasattr(
    model,
    "feature_names_in_"
):

    model_features = list(
        model.feature_names_in_
    )

else:

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
        "relative_recent_kills_diff"
    ]


# ============================================================
# PRINT MODEL FEATURES
# ============================================================

print("\nModel features:")

for feature in model_features:

    print(
        "-",
        feature
    )


# ============================================================
# CALCULATE ELO
# ============================================================

def calculate_elo(data):

    data = data.copy()

    teams = sorted(
        data["team"].unique()
    )

    elo = {

        team: 1000.0

        for team in teams

    }

    K = 32

    history = []

    match_numbers = sorted(
        data["match_num"].unique()
    )


    for match_num in match_numbers:

        match = data[
            data["match_num"]
            ==
            match_num
        ].copy()


        # ----------------------------------------------------
        # SAVE PRE-MATCH ELO
        # ----------------------------------------------------

        for _, row in match.iterrows():

            history.append({

                "team":
                    row["team"],

                "match_id":
                    row["match_id"],

                "match_num":
                    match_num,

                "pre_match_elo":
                    elo[row["team"]]

            })


        # ----------------------------------------------------
        # FIND WINNER
        # ----------------------------------------------------

        winners = match[
            match["win"]
            ==
            1
        ]


        if len(winners) != 1:

            continue


        winner = (
            winners
            .iloc[0]["team"]
        )


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


        # ----------------------------------------------------
        # UPDATE ELO
        # ----------------------------------------------------

        for team in match_teams:

            current_elo = (
                old_ratings[team]
            )


            expected = (

                1

                /

                (

                    1

                    +

                    10 ** (

                        (
                            average_elo
                            -
                            current_elo
                        )

                        /
                        400

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

                K

                *

                (
                    actual
                    -
                    expected
                )

            )


    return pd.DataFrame(
        history
    )


# ============================================================
# CREATE ELO DATA
# ============================================================

print("\nCalculating Elo ratings...")

elo_df = calculate_elo(
    raw_df
)

print("Elo ratings calculated successfully!")


# ============================================================
# MERGE ELO DATA
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
# CREATE ELO FEATURES
# ============================================================

mean_elo = (

    df

    .groupby(
        "match_num"
    )[
        "pre_match_elo"
    ]

    .transform(
        "mean"
    )

)


df["elo_difference"] = (

    df["pre_match_elo"]

    -

    mean_elo

)


df["elo_probability"] = (

    1

    /

    (

        1

        +

        10 ** (

            (

                mean_elo

                -

                df["pre_match_elo"]

            )

            /

            400

        )

    )

)


# ============================================================
# CHECK AVAILABLE COLUMNS
# ============================================================

print("\nAvailable columns:")

print(
    list(
        df.columns
    )
)


# ============================================================
# CREATE PAIRWISE EVALUATION DATA
# ============================================================

print("\nCreating evaluation pairs...")

evaluation_rows = []


match_numbers = sorted(
    df["match_num"].unique()
)


# ============================================================
# LOOP THROUGH MATCHES
# ============================================================

for match_num in match_numbers:


    # --------------------------------------------------------
    # FEATURE DATA
    # --------------------------------------------------------

    match_features = df[

        df["match_num"]
        ==
        match_num

    ].copy()


    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    match_raw = raw_df[

        raw_df["match_num"]
        ==
        match_num

    ].copy()


    # --------------------------------------------------------
    # FIND WINNER
    # --------------------------------------------------------

    winner_rows = match_raw[

        match_raw["win"]
        ==
        1

    ]


    if len(winner_rows) != 1:

        continue


    winner = (

        winner_rows
        .iloc[0]["team"]

    )


    # --------------------------------------------------------
    # TEAMS
    # --------------------------------------------------------

    teams_in_match = (

        match_features["team"]
        .tolist()

    )


    # ========================================================
    # CREATE PAIRS
    # ========================================================

    for i in range(
        len(teams_in_match)
    ):


        for j in range(
            i + 1,
            len(teams_in_match)
        ):


            team_a = (
                teams_in_match[i]
            )


            team_b = (
                teams_in_match[j]
            )


            # ------------------------------------------------
            # GET TEAM DATA
            # ------------------------------------------------

            data_a = match_features[

                match_features["team"]
                ==
                team_a

            ].iloc[0]


            data_b = match_features[

                match_features["team"]
                ==
                team_b

            ].iloc[0]


            # ------------------------------------------------
            # CREATE MODEL INPUT
            # ------------------------------------------------

            prediction_values = []


            valid_pair = True


            missing_features = []


            for feature in model_features:


                # --------------------------------------------
                # FEATURE MUST END WITH _diff
                # --------------------------------------------

                if not feature.endswith(
                    "_diff"
                ):

                    missing_features.append(
                        feature
                    )

                    valid_pair = False

                    continue


                # --------------------------------------------
                # GET BASE FEATURE
                # --------------------------------------------

                base_feature = (
                    feature[:-5]
                )


                # --------------------------------------------
                # CHECK FEATURE EXISTS
                # --------------------------------------------

                if (

                    base_feature
                    not in data_a.index

                    or

                    base_feature
                    not in data_b.index

                ):

                    missing_features.append(
                        base_feature
                    )

                    valid_pair = False

                    continue


                # --------------------------------------------
                # GET VALUES
                # --------------------------------------------

                value_a = float(

                    data_a[
                        base_feature
                    ]

                )


                value_b = float(

                    data_b[
                        base_feature
                    ]

                )


                # --------------------------------------------
                # CALCULATE DIFFERENCE
                # --------------------------------------------

                difference = (

                    value_a

                    -

                    value_b

                )


                prediction_values.append(
                    difference
                )


            # ------------------------------------------------
            # SKIP INVALID PAIRS
            # ------------------------------------------------

            if not valid_pair:

                continue


            # ------------------------------------------------
            # CHECK FEATURE COUNT
            # ------------------------------------------------

            if (

                len(prediction_values)

                !=

                len(model_features)

            ):

                continue


            # ------------------------------------------------
            # CREATE DATAFRAME
            # ------------------------------------------------

            X_pair = pd.DataFrame(

                [

                    prediction_values

                ],

                columns=model_features

            )


            # ------------------------------------------------
            # MODEL PROBABILITY
            # ------------------------------------------------

            probability = float(

                model.predict_proba(
                    X_pair
                )[0][1]

            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            prediction = (

                1

                if probability >= 0.5

                else 0

            )


            # ------------------------------------------------
            # ACTUAL RESULT
            # ------------------------------------------------

            actual = (

                1

                if team_a == winner

                else 0

            )


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            evaluation_rows.append({

                "match_num":
                    match_num,

                "team_a":
                    team_a,

                "team_b":
                    team_b,

                "winner":
                    winner,

                "actual":
                    actual,

                "prediction":
                    prediction,

                "probability":
                    probability

            })


# ============================================================
# CREATE EVALUATION DATAFRAME
# ============================================================

evaluation_df = pd.DataFrame(
    evaluation_rows
)


# ============================================================
# CHECK EVALUATION DATA
# ============================================================

print(
    f"\nTotal evaluation pairs created: "
    f"{len(evaluation_df)}"
)


if len(
    evaluation_df
) == 0:


    print("\nERROR:")

    print(
        "No evaluation pairs were created."
    )


    print(
        "\nPossible reason:"
    )

    print(
        "Model features do not match "
        "the available dataset features."
    )


else:


    # ========================================================
    # ACTUAL VALUES
    # ========================================================

    y_true = (

        evaluation_df[
            "actual"
        ]

    )


    # ========================================================
    # PREDICTED VALUES
    # ========================================================

    y_pred = (

        evaluation_df[
            "prediction"
        ]

    )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = (

        accuracy_score(
            y_true,
            y_pred
        )

    )


    # ========================================================
    # PRECISION
    # ========================================================

    precision = (

        precision_score(

            y_true,

            y_pred,

            zero_division=0

        )

    )


    # ========================================================
    # RECALL
    # ========================================================

    recall = (

        recall_score(

            y_true,

            y_pred,

            zero_division=0

        )

    )


    # ========================================================
    # F1 SCORE
    # ========================================================

    f1 = (

        f1_score(

            y_true,

            y_pred,

            zero_division=0

        )

    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = (

        confusion_matrix(

            y_true,

            y_pred

        )

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")

    print(
        "=" * 60
    )


    print(
        "MODEL EVALUATION RESULTS"
    )


    print(
        "=" * 60
    )


    print(

        f"\nTotal Pairwise Predictions: "

        f"{len(evaluation_df)}"

    )


    print(

        f"Accuracy: "

        f"{accuracy * 100:.2f}%"

    )


    print(

        f"Precision: "

        f"{precision * 100:.2f}%"

    )


    print(

        f"Recall: "

        f"{recall * 100:.2f}%"

    )


    print(

        f"F1 Score: "

        f"{f1 * 100:.2f}%"

    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        cm
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_df = pd.DataFrame({

        "Metric": [

            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"

        ],

        "Value": [

            accuracy,
            precision,
            recall,
            f1

        ]

    })


    metrics_df.to_csv(

        "data/model_metrics.csv",

        index=False

    )


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    confusion_df = pd.DataFrame(

        cm,

        index=[

            "Actual Loss",
            "Actual Win"

        ],

        columns=[

            "Predicted Loss",
            "Predicted Win"

        ]

    )


    confusion_df.to_csv(

        "data/confusion_matrix.csv"

    )


    # ========================================================
    # SAVE EVALUATION PREDICTIONS
    # ========================================================

    evaluation_df.to_csv(

        "data/model_evaluation_predictions.csv",

        index=False

    )


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    print("\n")

    print(
        "=" * 60
    )


    print(
        "FILES SAVED SUCCESSFULLY!"
    )


    print(
        "=" * 60
    )


    print(
        "\nCreated files:"
    )


    print(
        "1. data/model_metrics.csv"
    )


    print(
        "2. data/confusion_matrix.csv"
    )


    print(
        "3. data/model_evaluation_predictions.csv"
    )


    print(
        "\nModel evaluation completed successfully!"
    )