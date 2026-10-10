import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="BMPS Win Probability",
    page_icon="🎮",
    layout="wide"
)

MODEL_FILE = "models/bmps_win_model.pkl"
DATA_FILE = "data/bmps_features.csv"

features = [
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

st.title("🎮 BMPS Win Probability")
st.write("BMPS Esports Analytics and Win Probability Prediction")

try:
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)

    st.success("BMPS model loaded successfully.")

    match_number = st.number_input(
        "Enter Match Number",
        min_value=1,
        max_value=int(df["match"].max()),
        value=1
    )

    match_data = df[df["match"] == match_number].copy()

    if len(match_data) == 0:
        st.warning("No data available for this match.")

    else:
        X = match_data[features].fillna(0)

        probabilities = model.predict_proba(X)[:, 1]

        match_data["win_probability"] = probabilities

        total = match_data["win_probability"].sum()

        if total > 0:
            match_data["win_probability"] = (
                match_data["win_probability"] / total
            ) * 100

        match_data = match_data.sort_values(
            "win_probability",
            ascending=False
        )

        st.subheader(f"Match {match_number} Prediction")

        display_data = match_data[
            ["team", "rank", "win_probability"]
        ].copy()

        display_data["win_probability"] = (
            display_data["win_probability"].round(2)
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Win Probability")

        chart_data = display_data.set_index("team")

        st.bar_chart(
            chart_data["win_probability"]
        )

        st.metric(
            "Probability Total",
            f"{display_data['win_probability'].sum():.2f}%"
        )

except Exception as e:
    st.error(f"Error: {e}")