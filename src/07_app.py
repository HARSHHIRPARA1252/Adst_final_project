import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Esports Win Probability",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

# ==========================================================
# FEATURES
# ==========================================================

BASE_FEATURES = [
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

DEFAULT_FEATURES = BASE_FEATURES.copy()

# ==========================================================
# TOURNAMENT CONFIG
# ==========================================================

TOURNAMENTS = {
    "BGIS": {
        "raw": [
            DATA_DIR / "bgis.csv",
            DATA_DIR / "bgis_combined.csv"
        ],
        "features": [
            DATA_DIR / "bgis_features.csv"
        ],
        "model": [
            MODEL_DIR / "bgis_final_model_v6.pkl",
            MODEL_DIR / "bgis_pairwise_model.pkl",
            MODEL_DIR / "bgis_pairwise_model_v5.pkl"
        ]
    },

    "BMPS": {
        "raw": [
            DATA_DIR / "bmps_clean.csv",
            DATA_DIR / "bmps_2026.csv",
            DATA_DIR / "bmps_predictions.csv"
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

# ==========================================================
# MODERN UI DESIGN
# ==========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(102, 51, 153, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 15%,
                rgba(30, 80, 180, 0.16),
                transparent 28%
            ),
            #080b14;
        color: #f5f7ff;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0b0e19 0%,
                #0d1120 55%,
                #090c15 100%
            );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .sidebar-logo {
        text-align: center;
        padding: 10px 5px 25px 5px;
    }

    .sidebar-logo-icon {
        font-size: 52px;
        margin-bottom: 5px;
    }

    .sidebar-logo-title {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    .sidebar-logo-subtitle {
        color: #8992ad;
        font-size: 12px;
        margin-top: 5px;
        letter-spacing: 1.5px;
    }

    .sidebar-section {
        color: #8992ad;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .sidebar-footer {
        margin-top: 30px;
        padding: 18px;
        border-radius: 16px;
        background:
            linear-gradient(
                135deg,
                rgba(99,55,190,0.18),
                rgba(38,81,177,0.10)
            );
        border: 1px solid rgba(130,100,220,0.18);
    }

    .sidebar-footer-title {
        font-weight: 700;
        font-size: 14px;
    }

    .sidebar-footer-text {
        color: #8d96ae;
        font-size: 11px;
        line-height: 1.6;
        margin-top: 8px;
    }

    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 42px 48px;
        border-radius: 26px;
        margin-bottom: 30px;

        background:
            radial-gradient(
                circle at 85% 25%,
                rgba(70,110,255,0.32),
                transparent 35%
            ),
            radial-gradient(
                circle at 15% 80%,
                rgba(145,55,220,0.28),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #1a1238,
                #101d48
            );

        border: 1px solid rgba(155,130,255,0.25);
        box-shadow:
            0 25px 60px rgba(0,0,0,0.28);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -100px;
        top: -100px;
        border-radius: 50%;
        background: rgba(105,85,255,0.15);
        filter: blur(10px);
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: #cfd5ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 44px;
        line-height: 1.1;
        font-weight: 850;
        letter-spacing: -1px;
        margin-bottom: 14px;
    }

    .hero-title span {
        background:
            linear-gradient(
                90deg,
                #ffffff,
                #bca7ff,
                #79a7ff
            );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        max-width: 720px;
        color: #aeb7d0;
        font-size: 16px;
        line-height: 1.7;
    }

    .hero-tournament {
        margin-top: 22px;
        font-size: 14px;
        font-weight: 700;
        color: #d7d5ff;
    }

    /* =====================================================
       METRIC CARDS
       ===================================================== */

    .metric-card {
        min-height: 145px;
        padding: 24px;
        border-radius: 20px;

        background:
            linear-gradient(
                145deg,
                rgba(26,30,45,0.95),
                rgba(14,17,29,0.95)
            );

        border: 1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 15px 35px rgba(0,0,0,0.20);

        transition: 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(132,105,255,0.35);
    }

    .metric-label {
        color: #9ba4bb;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-icon {
        font-size: 20px;
        margin-right: 8px;
    }

    .metric-value {
        font-size: 38px;
        font-weight: 850;
        margin-top: 12px;
        color: #ffffff;
    }

    .metric-description {
        color: #6f7891;
        font-size: 11px;
        margin-top: 5px;
    }

    /* =====================================================
       SECTION
       ===================================================== */

    .section-title {
        font-size: 24px;
        font-weight: 800;
        margin-top: 28px;
        margin-bottom: 15px;
    }

    .section-subtitle {
        color: #8992aa;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* =====================================================
       PANEL
       ===================================================== */

    .panel {
        background:
            linear-gradient(
                145deg,
                rgba(21,25,38,0.96),
                rgba(11,14,24,0.96)
            );

        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* =====================================================
       PREDICTOR
       ===================================================== */

    .predictor-header {
        padding: 25px;
        border-radius: 20px;

        background:
            linear-gradient(
                135deg,
                rgba(86,53,177,0.18),
                rgba(32,75,170,0.12)
            );

        border: 1px solid rgba(120,100,220,0.18);
        margin-bottom: 22px;
    }

    .predictor-title {
        font-size: 27px;
        font-weight: 800;
    }

    .predictor-description {
        color: #929bb2;
        margin-top: 7px;
        font-size: 13px;
    }

    .probability-card {
        padding: 20px;
        border-radius: 18px;
        background:
            linear-gradient(
                145deg,
                #151927,
                #0e111b
            );
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 12px;
    }

    .probability-team {
        font-weight: 700;
        font-size: 15px;
    }

    .probability-number {
        font-size: 25px;
        font-weight: 850;
        margin-top: 5px;
    }

    /* =====================================================
       TABLE
       ===================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
    }

    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(135,110,255,0.35);
        background:
            linear-gradient(
                135deg,
                #5d3bd4,
                #315bc7
            );
        color: white;
        font-weight: 750;
        min-height: 48px;
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        border-color: #9c88ff;
        box-shadow:
            0 8px 25px rgba(79,58,190,0.35);
        transform: translateY(-1px);
    }

    /* =====================================================
       SELECT BOX
       ===================================================== */

    div[data-baseweb="select"] > div {
        background-color: #151925;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.09);
    }

    /* =====================================================
       INFO / SUCCESS / ERROR
       ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* =====================================================
       HIDE STREAMLIT BRANDING
       ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

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
            "total_points",
            "total_score"
        ],

        "kills": [
            "kills",
            "kill",
            "total_kills",
            "elim_points"
        ],

        "win": [
            "win",
            "wins",
            "winner",
            "match_winner"
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

        df = normalize_columns(df)

        df = add_match_num(df)

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


# ==========================================================
# ELO
# ==========================================================

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

    return pd.DataFrame(history)


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

    elo = calculate_elo(raw)

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


# ==========================================================
# MODEL PROBABILITY
# ==========================================================

def get_probability(
    model,
    X
):

    probability = model.predict_proba(X)

    if probability.shape[1] == 2:

        classes = list(
            getattr(
                model,
                "classes_",
                [0, 1]
            )
        )

        if 1 in classes:

            index = classes.index(1)

            return float(
                probability[0][index]
            )

        return float(
            probability[0][-1]
        )

    return float(
        probability[0][0]
    )


# ==========================================================
# PAIR PREDICTION
# ==========================================================

def predict_pair(
    model,
    row_a,
    row_b
):

    model_features = get_model_features(
        model
    )

    is_pairwise = any(
        feature.endswith("_diff")
        for feature in model_features
    )

    if not is_pairwise:

        try:

            values_a = []
            values_b = []

            for feature in model_features:

                if feature not in row_a.index:

                    return (
                        None,
                        f"Missing feature for Team A: {feature}"
                    )

                if feature not in row_b.index:

                    return (
                        None,
                        f"Missing feature for Team B: {feature}"
                    )

                value_a = pd.to_numeric(
                    pd.Series(
                        [row_a[feature]]
                    ),
                    errors="coerce"
                ).iloc[0]

                value_b = pd.to_numeric(
                    pd.Series(
                        [row_b[feature]]
                    ),
                    errors="coerce"
                ).iloc[0]

                if pd.isna(value_a):

                    return (
                        None,
                        f"Invalid Team A feature: {feature}"
                    )

                if pd.isna(value_b):

                    return (
                        None,
                        f"Invalid Team B feature: {feature}"
                    )

                values_a.append(
                    float(value_a)
                )

                values_b.append(
                    float(value_b)
                )

            X_a = pd.DataFrame(
                [values_a],
                columns=model_features
            )

            X_b = pd.DataFrame(
                [values_b],
                columns=model_features
            )

            probability_a = get_probability(
                model,
                X_a
            )

            probability_b = get_probability(
                model,
                X_b
            )

            total = (
                probability_a
                +
                probability_b
            )

            if total <= 0:

                return (
                    None,
                    "Invalid model probabilities."
                )

            probability_a = (
                probability_a
                /
                total
            )

            probability_a = max(
                0.0,
                min(
                    1.0,
                    probability_a
                )
            )

            return (
                probability_a,
                None
            )

        except Exception as e:

            return (
                None,
                str(e)
            )

    values = []

    for feature in model_features:

        if not feature.endswith("_diff"):

            return (
                None,
                f"Unsupported pairwise model feature: {feature}"
            )

        base = feature[:-5]

        if (
            base not in row_a.index
            or
            base not in row_b.index
        ):

            return (
                None,
                f"Missing feature: {base}"
            )

        value_a = pd.to_numeric(
            pd.Series(
                [row_a[base]]
            ),
            errors="coerce"
        ).iloc[0]

        value_b = pd.to_numeric(
            pd.Series(
                [row_b[base]]
            ),
            errors="coerce"
        ).iloc[0]

        if (
            pd.isna(value_a)
            or
            pd.isna(value_b)
        ):

            return (
                None,
                f"Invalid feature value: {base}"
            )

        values.append(
            float(value_a)
            -
            float(value_b)
        )

    X = pd.DataFrame(
        [values],
        columns=model_features
    )

    try:

        probability = get_probability(
            model,
            X
        )

        probability = max(
            0.0,
            min(
                1.0,
                probability
            )
        )

        return (
            probability,
            None
        )

    except Exception as e:

        return (
            None,
            str(e)
        )


# ==========================================================
# MAIN PREDICTION
# ==========================================================

def predict_match(
    model,
    match_features
):

    if model is None:

        return (
            None,
            "Trained model was not found."
        )

    if match_features.empty:

        return (
            None,
            "No feature data available for this match."
        )

    if "team" not in match_features.columns:

        return (
            None,
            "Team column is missing from feature dataset."
        )

    teams = (
        match_features["team"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if len(teams) == 0:

        return (
            None,
            "No teams found for this match."
        )

    model_features = get_model_features(
        model
    )

    is_pairwise = any(
        feature.endswith("_diff")
        for feature in model_features
    )

    results = []

    # ======================================================
    # NORMAL MODEL
    # ======================================================

    if not is_pairwise:

        probabilities = []

        for team in teams:

            row = match_features[
                match_features["team"]
                .astype(str)
                .str.strip()
                ==
                team
            ].iloc[0]

            values = []
            missing = []

            for feature in model_features:

                if feature not in row.index:

                    missing.append(
                        feature
                    )

                    continue

                value = pd.to_numeric(
                    pd.Series(
                        [row[feature]]
                    ),
                    errors="coerce"
                ).iloc[0]

                if pd.isna(value):

                    missing.append(
                        feature
                    )

                else:

                    values.append(
                        float(value)
                    )

            if missing:

                return (
                    None,
                    "Missing/invalid model features for "
                    f"{team}: "
                    +
                    ", ".join(missing)
                )

            X = pd.DataFrame(
                [values],
                columns=model_features
            )

            try:

                probability = get_probability(
                    model,
                    X
                )

            except Exception as e:

                return (
                    None,
                    f"Prediction failed for {team}: {e}"
                )

            probabilities.append(
                probability
            )

        probabilities = np.array(
            probabilities,
            dtype=float
        )

        probabilities = np.nan_to_num(
            probabilities,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        total = probabilities.sum()

        if total <= 0:

            probabilities = (
                np.ones(
                    len(teams)
                )
                /
                len(teams)
            )

        else:

            probabilities = (
                probabilities
                /
                total
            )

        for team, probability in zip(
            teams,
            probabilities
        ):

            results.append({
                "Team": team,
                "Win Probability":
                    probability * 100
            })

    # ======================================================
    # PAIRWISE MODEL
    # ======================================================

    else:

        for team_a in teams:

            row_a = match_features[
                match_features["team"]
                .astype(str)
                .str.strip()
                ==
                team_a
            ].iloc[0]

            pair_probabilities = []

            for team_b in teams:

                if team_a == team_b:
                    continue

                row_b = match_features[
                    match_features["team"]
                    .astype(str)
                    .str.strip()
                    ==
                    team_b
                ].iloc[0]

                probability, error = predict_pair(
                    model,
                    row_a,
                    row_b
                )

                if error:

                    return (
                        None,
                        f"{team_a} vs {team_b}: {error}"
                    )

                pair_probabilities.append(
                    probability
                )

            if pair_probabilities:

                average_probability = np.mean(
                    pair_probabilities
                )

            else:

                average_probability = 0.0

            results.append({
                "Team": team_a,
                "Win Probability":
                    average_probability * 100
            })

        total = sum(
            item["Win Probability"]
            for item in results
        )

        if total > 0:

            for item in results:

                item["Win Probability"] = (
                    item["Win Probability"]
                    /
                    total
                    *
                    100
                )

    prediction_df = pd.DataFrame(
        results
    )

    if prediction_df.empty:

        return (
            None,
            "Prediction result is empty."
        )

    prediction_df = (
        prediction_df
        .sort_values(
            "Win Probability",
            ascending=False
        )
        .reset_index(drop=True)
    )

    prediction_df.insert(
        0,
        "Rank",
        range(
            1,
            len(prediction_df) + 1
        )
    )

    return (
        prediction_df,
        None
    )


# ==========================================================
# MATCH HELPERS
# ==========================================================

def get_match_options(df):

    if (
        df.empty
        or
        "match_num" not in df.columns
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


# ==========================================================
# LEADERBOARD
# ==========================================================

def make_leaderboard(raw):

    if (
        raw.empty
        or
        "team" not in raw.columns
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


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown(
    """
    <div class="sidebar-logo">

        <div class="sidebar-logo-icon">
            🎮
        </div>

        <div class="sidebar-logo-title">
            ESPORTS ANALYTICS
        </div>

        <div class="sidebar-logo-subtitle">
            WIN PROBABILITY SYSTEM
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-section">Tournament</div>',
    unsafe_allow_html=True
)

tournament = st.sidebar.selectbox(
    "🏆 Select Tournament",
    [
        "BGIS",
        "BMPS",
        "BMSD"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown(
    '<div class="sidebar-section">Navigation</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Match Predictor",
        "📜 Match History",
        "🏆 Leaderboard",
        "📊 Team Analytics",
        "🤖 Model Analytics"
    ],
    label_visibility="collapsed"
)

# ==========================================================
# LOAD DATA
# ==========================================================

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

# ==========================================================
# SIDEBAR FOOTER
# ==========================================================

st.sidebar.markdown(
    """
    <div class="sidebar-footer">

        <div class="sidebar-footer-title">
            🎮 Esports Win Probability
        </div>

        <div class="sidebar-footer-text">
            AI-powered esports analytics
            for tournament performance,
            team statistics and match
            win probability.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# ERROR IF DATA NOT FOUND
# ==========================================================

if raw.empty:

    st.error(
        f"{tournament} data file was not found."
    )

    st.code(
        str(DATA_DIR)
    )

    st.stop()

# ==========================================================
# HERO
# ==========================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="hero-badge">
            🏆 {tournament} ANALYTICS
        </div>

        <div class="hero-title">
            🎮 <span>ESPORTS WIN PROBABILITY</span>
        </div>

        <div class="hero-subtitle">
            AI-powered esports analytics,
            team performance analysis and
            match win probability prediction.
        </div>

        <div class="hero-tournament">
            ● LIVE DATA • {tournament} • PERFORMANCE INTELLIGENCE
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# METRICS
# ==========================================================

matches = (
    raw["match_num"].nunique()
    if "match_num" in raw.columns
    else 0
)

teams = (
    raw["team"].nunique()
    if "team" in raw.columns
    else 0
)

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

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                <span class="metric-icon">🏁</span>
                Matches
            </div>

            <div class="metric-value">
                {matches}
            </div>

            <div class="metric-description">
                Tournament matches
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with m2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                <span class="metric-icon">👥</span>
                Teams
            </div>

            <div class="metric-value">
                {teams}
            </div>

            <div class="metric-description">
                Participating teams
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with m3:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                <span class="metric-icon">🎯</span>
                Total Points
            </div>

            <div class="metric-value">
                {int(points)}
            </div>

            <div class="metric-description">
                Combined tournament points
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with m4:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                <span class="metric-icon">💀</span>
                Total Kills
            </div>

            <div class="metric-value">
                {int(kills)}
            </div>

            <div class="metric-description">
                Combined eliminations
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# DASHBOARD
# ==========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">🏠 Tournament Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="section-subtitle">
            Overview of {tournament} team performance and tournament statistics.
        </div>
        """,
        unsafe_allow_html=True
    )

    board = make_leaderboard(
        raw
    )

    if board.empty:

        st.warning(
            "No team data available."
        )

    else:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🏆 Team Performance</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            board,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    if "points" in raw.columns:

        st.markdown(
            '<div class="section-title">📈 Average Points by Team</div>',
            unsafe_allow_html=True
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


# ==========================================================
# MATCH PREDICTOR
# ==========================================================

elif page == "🔮 Match Predictor":

    st.markdown(
        """
        <div class="predictor-header">

            <div class="predictor-title">
                🔮 Match Win Probability
            </div>

            <div class="predictor-description">
                Select a tournament match and use the trained
                machine learning model to calculate team win probabilities.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if tournament == "BMSD":

        st.info(
            "BMSD analytics is available. "
            "A prediction model can be added after "
            "BMSD model training is completed."
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

        match_options = get_match_options(
            features
        )

        if not match_options:

            st.error(
                "No match numbers were found "
                "in the feature dataset."
            )

        else:

            selected_match = st.selectbox(
                "🎯 Select Match",
                match_options,
                key=f"{tournament}_prediction_match"
            )

            selected_number = parse_match(
                selected_match
            )

            match_features = features[
                features["match_num"]
                ==
                selected_number
            ].copy()

            st.caption(
                f"📍 {tournament} • {selected_match}"
            )

            if not match_features.empty:

                if st.button(
                    "🔮 Predict Win Probability",
                    use_container_width=True
                ):

                    prediction_df, error = predict_match(
                        model,
                        match_features
                    )

                    if error:

                        st.error(
                            error
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="section-title">
                                📊 {selected_match} Win Probability
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        display_df = prediction_df[
                            [
                                "Rank",
                                "Team",
                                "Win Probability"
                            ]
                        ].copy()

                        display_df[
                            "Win Probability"
                        ] = (
                            display_df[
                                "Win Probability"
                            ].round(2)
                        )

                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.markdown(
                            '<div class="section-title">📈 Probability Chart</div>',
                            unsafe_allow_html=True
                        )

                        chart = (
                            prediction_df
                            .set_index("Team")
                            ["Win Probability"]
                        )

                        st.bar_chart(
                            chart
                        )

                        top_team = (
                            prediction_df.iloc[0]["Team"]
                        )

                        top_probability = float(
                            prediction_df.iloc[0]
                            ["Win Probability"]
                        )

                        st.success(
                            f"🏆 Highest model probability: "
                            f"{top_team} "
                            f"({top_probability:.2f}%)"
                        )

                        if "win" in match_features.columns:

                            actual = match_features[
                                match_features["win"] == 1
                            ]

                            if not actual.empty:

                                actual_team = str(
                                    actual.iloc[0]["team"]
                                )

                                st.info(
                                    f"📌 Historical result: {actual_team}"
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


# ==========================================================
# MATCH HISTORY
# ==========================================================

elif page == "📜 Match History":

    st.markdown(
        '<div class="section-title">📜 Match History</div>',
        unsafe_allow_html=True
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
            "🎯 Select Match",
            options,
            key=f"{tournament}_history_match"
        )

        number = parse_match(
            selected
        )

        match_data = raw[
            raw["match_num"]
            ==
            number
        ].copy()

        st.markdown(
            f"""
            <div class="panel">

                <div class="section-title">
                    🏁 {selected}
                </div>

            </div>
            """,
            unsafe_allow_html=True
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
                        winners.iloc[0]["team"]
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

            if "points" in match_data.columns:

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


# ==========================================================
# LEADERBOARD
# ==========================================================

elif page == "🏆 Leaderboard":

    st.markdown(
        '<div class="section-title">🏆 Tournament Leaderboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="section-subtitle">
            Overall {tournament} team statistics.
        </div>
        """,
        unsafe_allow_html=True
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


# ==========================================================
# TEAM ANALYTICS
# ==========================================================

elif page == "📊 Team Analytics":

    st.markdown(
        '<div class="section-title">📊 Team Analytics</div>',
        unsafe_allow_html=True
    )

    teams_list = sorted(
        raw["team"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not teams_list:

        st.warning(
            "No teams found."
        )

    else:

        selected_team = st.selectbox(
            "👥 Select Team",
            teams_list
        )

        team_data = raw[
            raw["team"]
            .astype(str)
            ==
            selected_team
        ].copy()

        t1, t2, t3 = st.columns(3)

        with t1:

            st.metric(
                "🏁 Matches",
                len(team_data)
            )

        with t2:

            total_points = 0

            if "points" in team_data.columns:

                total_points = (
                    pd.to_numeric(
                        team_data["points"],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

            st.metric(
                "🎯 Points",
                int(total_points)
            )

        with t3:

            total_kills = 0

            if "kills" in team_data.columns:

                total_kills = (
                    pd.to_numeric(
                        team_data["kills"],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

            st.metric(
                "💀 Kills",
                int(total_kills)
            )

        if (
            "points" in team_data.columns
            and
            "kills" in team_data.columns
        ):

            st.markdown(
                '<div class="section-title">📈 Match Performance</div>',
                unsafe_allow_html=True
            )

            st.line_chart(
                team_data[
                    [
                        "points",
                        "kills"
                    ]
                ]
            )


# ==========================================================
# MODEL ANALYTICS
# ==========================================================

elif page == "🤖 Model Analytics":

    st.markdown(
        '<div class="section-title">🤖 Model Analytics</div>',
        unsafe_allow_html=True
    )

    if model is None:

        st.info(
            "No trained model is configured "
            "for this tournament."
        )

    else:

        model_features = get_model_features(
            model
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        🤖 Model Type
                    </div>

                    <div class="metric-value"
                         style="font-size:25px;">
                        {type(model).__name__}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        🧠 Features
                    </div>

                    <div class="metric-value">
                        {len(model_features)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section-title">🧠 Model Features</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            pd.DataFrame({
                "Feature": model_features
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

                coefficient_df = pd.DataFrame({
                    "Feature":
                        model_features,
                    "Coefficient":
                        coefficients
                })

                coefficient_df[
                    "Absolute Impact"
                ] = (
                    coefficient_df[
                        "Coefficient"
                    ].abs()
                )

                st.markdown(
                    '<div class="section-title">📈 Feature Impact</div>',
                    unsafe_allow_html=True
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

        st.markdown(
            '<div class="section-title">📁 Dataset Information</div>',
            unsafe_allow_html=True
        )

        d1, d2, d3 = st.columns(3)

        with d1:

            st.metric(
                "Raw Matches",
                (
                    raw["match_num"].nunique()
                    if "match_num" in raw.columns
                    else 0
                )
            )

        with d2:

            st.metric(
                "Feature Matches",
                (
                    features["match_num"].nunique()
                    if (
                        not features.empty
                        and
                        "match_num" in features.columns
                    )
                    else 0
                )
            )

        with d3:

            st.metric(
                "Teams",
                (
                    raw["team"].nunique()
                    if "team" in raw.columns
                    else 0
                )
            )

# ==========================================================
# END
# ==========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:60px;
        padding-top:25px;
        border-top:1px solid rgba(255,255,255,0.06);
        color:#69728b;
        font-size:12px;
    ">
        🎮 Esports Win Probability System
        <br>
        BGIS • BMPS • BMSD
        <br>
        AI-Powered Esports Analytics
    </div>
    """,
    unsafe_allow_html=True
)