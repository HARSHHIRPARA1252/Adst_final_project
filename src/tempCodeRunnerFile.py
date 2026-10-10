import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Esports Win Probability System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

DEFAULT_FEATURES = [
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

TOURNAMENTS = {
    "BGIS": {
        "raw": [
            DATA_DIR / "bgis.csv"
        ],
        "features": [
            DATA_DIR / "bgis_features.csv"
        ],
        "model": [
            MODEL_DIR / "bgis_final_model_v6.pkl"
        ]
    },

    "BMPS": {
        "raw": [
            DATA_DIR / "bmps_clean.csv",
            DATA_DIR / "bmps_predictions.csv",
            DATA_DIR / "bmps_2026.csv"
        ],
        "features": [
            DATA_DIR / "bmps_features.csv"
        ],
        "model": [
            MODEL_DIR / "bmps_win_model.pkl"
        ]
    },

    "BMSD": {
        "raw": [
            DATA_DIR / "bmsd.csv"
        ],
        "features": [],
        "model": []
    }
}


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .big-number {
        font-size: 34px;
        font-weight: 800;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def first_existing(paths):

    for path in paths:

        if path.exists():
            return path

    return None


def normalize_columns(df):

    df = df.copy()

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    aliases = {

        "team": [
            "team",
            "team_name",
            "team name"
        ],

        "match_id": [
            "match_id",
            "match",
            "match id"
        ],

        "points": [
            "points",
            "point",
            "total_points"
        ],

        "kills": [
            "kills",
            "kill",
            "total_kills"
        ],

        "win": [
            "win",
            "wins",
            "winner"
        ]
    }

    rename = {}

    for target, names in aliases.items():

        for name in names:

            key = name.lower()

            if key in lookup:

                rename[
                    lookup[key]
                ] = target

                break

    return df.rename(
        columns=rename
    )


def add_match_num(df):

    df = df.copy()

    if "match_num" in df.columns:

        df["match_num"] = pd.to_numeric(
            df["match_num"],
            errors="coerce"
        )

        return df

    source = None

    for name in [
        "match_id",
        "match",
        "Match",
        "Match ID"
    ]:

        if name in df.columns:

            source = name
            break

    if source is None:
        return df

    df["match_num"] = pd.to_numeric(
        df[source]
        .astype(str)
        .str.strip()
        .str.replace(
            "M",
            "",
            regex=False
        )
        .str.replace(
            "m",
            "",
            regex=False
        ),
        errors="coerce"
    )

    return df


def convert_win(value):

    text = str(value).strip().lower()

    if text in [
        "1",
        "true",
        "yes",
        "win",
        "winner"
    ]:
        return 1

    try:

        return (
            1
            if float(value) == 1
            else 0
        )

    except Exception:

        return 0


@st.cache_data(show_spinner=False)
def load_csv(path):

    if (
        path is None
        or not path.exists()
    ):
        return pd.DataFrame()

    try:

        df = pd.read_csv(path)

        df = normalize_columns(
            df
        )

        df = add_match_num(
            df
        )

        if "team" in df.columns:

            df["team"] = (
                df["team"]
                .astype(str)
                .str.strip()
            )

        if "win" in df.columns:

            df["win"] = (
                df["win"]
                .apply(convert_win)
            )

        return df

    except Exception:

        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model(path):

    if (
        path is None
        or not path.exists()
    ):
        return None

    try:

        return joblib.load(path)

    except Exception:

        return None


def get_model_features(model):

    if (
        model is not None
        and hasattr(
            model,
            "feature_names_in_"
        )
    ):

        return list(
            model.feature_names_in_
        )

    return DEFAULT_FEATURES.copy()


def calculate_elo(raw):

    if raw.empty:
        return pd.DataFrame()

    required = {
        "team",
        "match_num",
        "win"
    }

    if not required.issubset(
        raw.columns
    ):
        return pd.DataFrame()

    teams = sorted(
        raw["team"]
        .dropna()
        .astype(str)
        .unique()
    )

    elo = {
        team: 1000.0
        for team in teams
    }

    history = []

    match_numbers = sorted(
        raw["match_num"]
        .dropna()
        .unique()
    )

    for match_num in match_numbers:

        match = raw[
            raw["match_num"]
            == match_num
        ].copy()

        match_teams = list(
            match["team"]
            .dropna()
            .astype(str)
            .unique()
        )

        for team in match_teams:

            history.append({
                "team": team,
                "match_num": match_num,
                "pre_match_elo":
                    elo.get(
                        team,
                        1000.0
                    )
            })

        winners = match[
            match["win"] == 1
        ]

        if len(winners) != 1:
            continue

        winner = str(
            winners.iloc[0]["team"]
        )

        old = {
            team: elo.get(
                team,
                1000.0
            )
            for team in match_teams
        }

        average_elo = np.mean(
            list(
                old.values()
            )
        )

        for team in match_teams:

            current = old[team]

            expected = (
                1
                /
                (
                    1
                    +
                    10 ** (
                        (
                            average_elo
                            - current
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
                current
                +
                32 * (
                    actual
                    - expected
                )
            )

    return pd.DataFrame(
        history
    )


def add_elo_features(
    raw,
    features
):

    if (
        raw.empty
        or features.empty
    ):
        return features

    if not {
        "team",
        "match_num"
    }.issubset(
        features.columns
    ):
        return features

    elo = calculate_elo(
        raw
    )

    if elo.empty:
        return features

    out = features.copy()

    for column in [
        "pre_match_elo",
        "elo_difference",
        "elo_probability"
    ]:

        if column in out.columns:

            out = out.drop(
                columns=[column]
            )

    out = out.merge(
        elo,
        on=[
            "team",
            "match_num"
        ],
        how="left"
    )

    mean_elo = (
        out.groupby(
            "match_num"
        )[
            "pre_match_elo"
        ].transform("mean")
    )

    out["elo_difference"] = (
        out["pre_match_elo"]
        -
        mean_elo
    )

    out["elo_probability"] = (
        1
        /
        (
            1
            +
            10 ** (
                (
                    mean_elo
                    -
                    out["pre_match_elo"]
                ) / 400
            )
        )
    )

    return out


def predict_pair(model, row_a, row_b):

    model_features = get_model_features(model)

    # -------------------------------------------------
    # BMPS MODEL
    # Model uses normal features, not _diff features
    # -------------------------------------------------

    if not any(
        feature.endswith("_diff")
        for feature in model_features
    ):

        try:

            values_a = []

            values_b = []

            for feature in model_features:

                if feature not in row_a.index:
                    return None, f"Missing feature for Team A: {feature}"

                if feature not in row_b.index:
                    return None, f"Missing feature for Team B: {feature}"

                value_a = pd.to_numeric(
                    pd.Series([row_a[feature]]),
                    errors="coerce"
                ).iloc[0]

                value_b = pd.to_numeric(
                    pd.Series([row_b[feature]]),
                    errors="coerce"
                ).iloc[0]

                if pd.isna(value_a):
                    return None, f"Invalid Team A feature: {feature}"

                if pd.isna(value_b):
                    return None, f"Invalid Team B feature: {feature}"

                values_a.append(float(value_a))

                values_b.append(float(value_b))

            X_a = pd.DataFrame(
                [values_a],
                columns=model_features
            )

            X_b = pd.DataFrame(
                [values_b],
                columns=model_features
            )

            probability_a = float(
                model.predict_proba(X_a)[0][1]
            )

            probability_b = float(
                model.predict_proba(X_b)[0][1]
            )

            total = probability_a + probability_b

            if total <= 0:
                return None, "Invalid model probabilities."

            probability_a = probability_a / total

            probability_a = max(
                0.0,
                min(1.0, probability_a)
            )

            return probability_a, None

        except Exception as e:

            return None, str(e)

    # -------------------------------------------------
    # BGIS MODEL
    # Model uses pairwise _diff features
    # -------------------------------------------------

    values = []

    for feature in model_features:

        if not feature.endswith("_diff"):

            return None, (
                f"Unsupported pairwise model feature: {feature}"
            )

        base = feature[:-5]

        if (
            base not in row_a.index
            or base not in row_b.index
        ):

            return None, (
                f"Missing feature: {base}"
            )

        value_a = pd.to_numeric(
            pd.Series([row_a[base]]),
            errors="coerce"
        ).iloc[0]

        value_b = pd.to_numeric(
            pd.Series([row_b[base]]),
            errors="coerce"
        ).iloc[0]

        if pd.isna(value_a) or pd.isna(value_b):

            return None, (
                f"Invalid feature value: {base}"
            )

        values.append(
            float(value_a) - float(value_b)
        )

    X = pd.DataFrame(
        [values],
        columns=model_features
    )

    try:

        probability = float(
            model.predict_proba(X)[0][1]
        )

        probability = max(
            0.0,
            min(1.0, probability)
        )

        return probability, None

    except Exception as e:

        return None, str(e)

def get_match_options(df):

    if (
        df.empty
        or
        "match_num"
        not in df.columns
    ):
        return []

    numbers = sorted(
        df["match_num"]
        .dropna()
        .unique()
    )

    return [
        (
            f"M{int(number)}"
            if float(number).is_integer()
            else f"M{number}"
        )
        for number in numbers
    ]


def parse_match(value):

    try:

        return float(
            str(value)
            .upper()
            .replace(
                "M",
                ""
            )
            .strip()
        )

    except Exception:

        return None


def make_leaderboard(raw):

    if (
        raw.empty
        or
        "team"
        not in raw.columns
    ):

        return pd.DataFrame()

    data = raw.copy()

    data["team"] = (
        data["team"]
        .astype(str)
        .str.strip()
    )

    board = (
        data
        .groupby("team")
        .size()
        .rename("Matches")
        .to_frame()
    )

    if "points" in data.columns:

        points = pd.to_numeric(
            data["points"],
            errors="coerce"
        ).fillna(0)

        board["Points"] = (
            points
            .groupby(data["team"])
            .sum()
        )

    if "kills" in data.columns:

        kills = pd.to_numeric(
            data["kills"],
            errors="coerce"
        ).fillna(0)

        board["Kills"] = (
            kills
            .groupby(data["team"])
            .sum()
        )

    if "win" in data.columns:

        board["Wins"] = (
            data["win"]
            .groupby(data["team"])
            .sum()
        )

        board["Win Rate"] = (
            board["Wins"]
            /
            board["Matches"]
            *
            100
        ).round(2)

    sort_columns = [
        column
        for column in [
            "Points",
            "Wins",
            "Kills"
        ]
        if column in board.columns
    ]

    if sort_columns:

        board = board.sort_values(
            sort_columns,
            ascending=False
        )

    board = board.reset_index()

    board.insert(
        0,
        "Rank",
        range(
            1,
            len(board) + 1
        )
    )

    return board


st.sidebar.title(
    "🎮 Esports Analytics"
)

st.sidebar.caption(
    "BGIS • BMPS • BMSD"
)

tournament = st.sidebar.selectbox(
    "Select Tournament",
    [
        "BGIS",
        "BMPS",
        "BMSD"
    ]
)

config = TOURNAMENTS[
    tournament
]

raw_path = first_existing(
    config["raw"]
)

features_path = first_existing(
    config["features"]
)

model_path = first_existing(
    config["model"]
)

raw = load_csv(
    raw_path
)

features = load_csv(
    features_path
)

model = load_model(
    model_path
)

if (
    not raw.empty
    and
    not features.empty
    and
    tournament in [
        "BGIS",
        "BMPS"
    ]
):

    features = add_elo_features(
        raw,
        features
    )


st.markdown(
    '<div class="main-title">'
    '🎮 ESPORTS WIN PROBABILITY SYSTEM'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'BGIS • BMPS • BMSD'
    '</div>',
    unsafe_allow_html=True
)


if raw.empty:

    st.error(
        f"{tournament} data file was not found."
    )

    st.code(
        str(DATA_DIR)
    )

    st.stop()


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "🏁 Matches",
        (
            raw["match_num"]
            .nunique()
            if "match_num"
            in raw.columns
            else 0
        )
    )

with c2:

    st.metric(
        "👥 Teams",
        (
            raw["team"]
            .nunique()
            if "team"
            in raw.columns
            else 0
        )
    )

with c3:

    points = 0

    if "points" in raw.columns:

        points = (
            pd.to_numeric(
                raw["points"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    st.metric(
        "🎯 Points",
        int(points)
    )

with c4:

    kills = 0

    if "kills" in raw.columns:

        kills = (
            pd.to_numeric(
                raw["kills"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    st.metric(
        "💀 Kills",
        int(kills)
    )


st.divider()


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Match Predictor",
        "📜 Match History",
        "🏆 Leaderboard",
        "📊 Team Analytics",
        "🤖 Model Analytics"
    ]
)


if page == "🏠 Dashboard":

    st.header(
        f"🏠 {tournament} Dashboard"
    )

    board = make_leaderboard(
        raw
    )

    if board.empty:

        st.warning(
            "No team data available."
        )

    else:

        st.subheader(
            "🏆 Team Rankings"
        )

        st.dataframe(
            board,
            use_container_width=True,
            hide_index=True
        )

    if "points" in raw.columns:

        st.subheader(
            "📈 Average Points"
        )

        average_points = (
            pd.to_numeric(
                raw["points"],
                errors="coerce"
            )
            .fillna(0)
            .groupby(
                raw["team"]
            )
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            average_points
        )


elif page == "🔮 Match Predictor":

    st.header(
        f"🔮 {tournament} Match Predictor"
    )

    if tournament == "BMSD":

        st.info(
            "BMSD analytics is available. "
            "A BMSD prediction model can "
            "be added later."
        )

    elif model is None:

        st.error(
            "Trained model was not found."
        )

        st.code(
            str(model_path)
            if model_path
            else
            "Model file not found."
        )

    elif features.empty:

        st.error(
            "Feature dataset was not found."
        )

        st.code(
            str(features_path)
            if features_path
            else
            "Feature file not found."
        )

    else:

        match_options = (
            get_match_options(
                features
            )
        )

        if not match_options:

            st.error(
                "No match numbers were found "
                "in the feature dataset."
            )

        else:

            selected_match = (
                st.selectbox(
                    "Select Match",
                    match_options,
                    key=(
                        f"{tournament}_"
                        "prediction_match"
                    )
                )
            )

            selected_number = (
                parse_match(
                    selected_match
                )
            )

            match_features = (
                features[
                    features["match_num"]
                    ==
                    selected_number
                ].copy()
            )

            st.caption(
                f"{tournament} • "
                f"{selected_match}"
            )

            if match_features.empty:

                st.error(
                    "No feature data available "
                    f"for {selected_match}."
                )

            else:

                if st.button(
                    "🔮 Predict Win Probability",
                    use_container_width=True
                ):

                    prediction_df, error = (
                        predict_match(
                            model,
                            match_features
                        )
                    )

                    if error:

                        st.error(
                            error
                        )

                    else:

                        st.subheader(
                            f"📊 "
                            f"{selected_match} "
                            "Win Probability"
                        )

                        display_df = (
                            prediction_df[
                                [
                                    "Rank",
                                    "Team",
                                    "Win Probability"
                                ]
                            ].copy()
                        )

                        display_df[
                            "Win Probability"
                        ] = display_df[
                            "Win Probability"
                        ].round(2)

                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.subheader(
                            "📈 Probability Chart"
                        )

                        chart = (
                            prediction_df
                            .set_index("Team")
                            [
                                "Win Probability"
                            ]
                        )

                        st.bar_chart(
                            chart
                        )

                        top_team = (
                            prediction_df
                            .iloc[0]["Team"]
                        )

                        top_probability = float(
                            prediction_df
                            .iloc[0]
                            ["Win Probability"]
                        )

                        st.success(
                            f"🏆 Highest model "
                            f"probability: "
                            f"{top_team} "
                            f"({top_probability:.2f}%)"
                        )

                        if "win" in (
                            match_features
                            .columns
                        ):

                            actual = (
                                match_features[
                                    match_features[
                                        "win"
                                    ] == 1
                                ]
                            )

                            if not actual.empty:

                                actual_team = str(
                                    actual.iloc[0]
                                    ["team"]
                                )

                                st.info(
                                    "Historical result: "
                                    +
                                    actual_team
                                )

                        csv_data = (
                            prediction_df
                            .to_csv(
                                index=False
                            )
                            .encode("utf-8")
                        )

                        st.download_button(
                            "📥 Download Prediction",
                            data=csv_data,
                            file_name=(
                                f"{tournament}_"
                                f"{selected_match}_"
                                "prediction.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )


elif page == "📜 Match History":

    st.header(
        f"📜 {tournament} Match History"
    )

    options = get_match_options(
        raw
    )

    if not options:

        st.warning(
            "No match data available."
        )

    else:

        selected = st.selectbox(
            "Select Match",
            options,
            key=(
                f"{tournament}_"
                "history_match"
            )
        )

        number = parse_match(
            selected
        )

        match_data = raw[
            raw["match_num"]
            ==
            number
        ].copy()

        st.subheader(
            selected
        )

        if "win" in match_data.columns:

            winners = match_data[
                match_data["win"] == 1
            ]

            if not winners.empty:

                st.success(
                    "🏆 Winner: "
                    +
                    str(
                        winners.iloc[0]
                        ["team"]
                    )
                )

        columns = [
            column
            for column in [
                "match_id",
                "team",
                "points",
                "kills",
                "win"
            ]
            if column in match_data.columns
        ]

        if columns:

            if "points" in (
                match_data.columns
            ):

                match_data = (
                    match_data
                    .sort_values(
                        "points",
                        ascending=False
                    )
                )

            st.dataframe(
                match_data[columns],
                use_container_width=True,
                hide_index=True
            )


elif page == "🏆 Leaderboard":

    st.header(
        f"🏆 {tournament} Leaderboard"
    )

    board = make_leaderboard(
        raw
    )

    if board.empty:

        st.warning(
            "No leaderboard data."
        )

    else:

        st.dataframe(
            board,
            use_container_width=True,
            hide_index=True
        )


elif page == "📊 Team Analytics":

    st.header(
        f"📊 {tournament} Team Analytics"
    )

    teams = sorted(
        raw["team"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not teams:

        st.warning(
            "No teams found."
        )

    else:

        selected_team = (
            st.selectbox(
                "Select Team",
                teams
            )
        )

        team_data = raw[
            raw["team"].astype(str)
            ==
            selected_team
        ].copy()

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Matches",
                len(team_data)
            )

        with c2:

            total_points = 0

            if "points" in (
                team_data.columns
            ):

                total_points = (
                    pd.to_numeric(
                        team_data[
                            "points"
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

            st.metric(
                "Points",
                int(total_points)
            )

        with c3:

            total_kills = 0

            if "kills" in (
                team_data.columns
            ):

                total_kills = (
                    pd.to_numeric(
                        team_data[
                            "kills"
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

            st.metric(
                "Kills",
                int(total_kills)
            )

        if (
            "points"
            in team_data.columns
            and
            "kills"
            in team_data.columns
        ):

            st.line_chart(
                team_data[
                    [
                        "points",
                        "kills"
                    ]
                ]
            )


elif page == "🤖 Model Analytics":

    st.header(
        f"🤖 {tournament} Model Analytics"
    )

    if model is None:

        st.info(
            "No trained model is configured "
            "for this tournament."
        )

    else:

        model_features = (
            get_model_features(
                model
            )
        )

        st.subheader(
            "Model Type"
        )

        st.write(
            type(model).__name__
        )

        st.subheader(
            "Model Features"
        )

        st.dataframe(
            pd.DataFrame({
                "Feature":
                    model_features
            }),
            use_container_width=True,
            hide_index=True
        )

        if hasattr(
            model,
            "coef_"
        ):

            coefficients = (
                np.asarray(
                    model.coef_
                ).reshape(-1)
            )

            if len(coefficients) == len(
                model_features
            ):

                coefficient_df = (
                    pd.DataFrame({
                        "Feature":
                            model_features,
                        "Coefficient":
                            coefficients
                    })
                )

                coefficient_df[
                    "Absolute Impact"
                ] = (
                    coefficient_df[
                        "Coefficient"
                    ].abs()
                )

                st.subheader(
                    "📈 Feature Impact"
                )

                st.dataframe(
                    coefficient_df
                    .sort_values(
                        "Absolute Impact",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        st.subheader(
            "📁 Dataset Information"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Raw Matches",
                (
                    raw["match_num"]
                    .nunique()
                    if "match_num"
                    in raw.columns
                    else 0
                )
            )

        with c2:

            st.metric(
                "Feature Matches",
                (
                    features[
                        "match_num"
                    ].nunique()
                    if (
                        not features.empty
                        and
                        "match_num"
                        in features.columns
                    )
                    else 0
                )
            )

        with c3:

            st.metric(
                "Teams",
                (
                    raw["team"]
                    .nunique()
                    if "team"
                    in raw.columns
                    else 0
                )
            )


st.sidebar.divider()

st.sidebar.info(
    "🎮 Esports Win Probability System\n\n"
    "BGIS + BMPS + BMSD\n\n"
    "BGIS: V6 Logistic Regression\n"
    "BMPS: BMPS Model\n"
    "BMSD: Analytics Data"
)