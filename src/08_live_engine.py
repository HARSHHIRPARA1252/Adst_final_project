import streamlit as st
import pandas as pd
import os

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="Live Win Probability",
    page_icon="🔴",
    layout="wide"
)

# ---------------------------------------
# LOAD DATA
# ---------------------------------------

@st.cache_data
def load_data():

    possible_files = [

        "data/features.csv",
        "data/processed_data.csv",
        "data/bgis_data.csv",
        "data/matches.csv"

    ]

    for file in possible_files:

        if os.path.exists(file):

            return pd.read_csv(file)

    return None


df = load_data()


# ---------------------------------------
# TITLE
# ---------------------------------------

st.title("🔴 Live Win Probability")

st.write(
    "Compare two BGIS teams using live points and kills."
)


# ---------------------------------------
# CHECK DATA
# ---------------------------------------

if df is None:

    st.error("❌ Dataset not found.")

    st.stop()


# ---------------------------------------
# FIND TEAM COLUMN
# ---------------------------------------

team_col = None

possible_team_columns = [

    "team",
    "Team",
    "team_name",
    "Team Name"

]


for column in possible_team_columns:

    if column in df.columns:

        team_col = column

        break


if team_col is None:

    st.error("❌ Team column not found.")

    st.write(df.columns.tolist())

    st.stop()


# ---------------------------------------
# FIND POINTS COLUMN
# ---------------------------------------

points_col = None

possible_points_columns = [

    "points",
    "Points",
    "total_points",
    "Total Points"

]


for column in possible_points_columns:

    if column in df.columns:

        points_col = column

        break


# ---------------------------------------
# FIND KILLS COLUMN
# ---------------------------------------

kills_col = None

possible_kills_columns = [

    "kills",
    "Kills",
    "total_kills",
    "Total Kills"

]


for column in possible_kills_columns:

    if column in df.columns:

        kills_col = column

        break


# ---------------------------------------
# GET TEAM LIST
# ---------------------------------------

teams = sorted(
    df[team_col]
    .dropna()
    .unique()
)


# ---------------------------------------
# SELECT TEAMS
# ---------------------------------------

col1, col2 = st.columns(2)


with col1:

    team_a = st.selectbox(

        "Team A",

        teams,

        key="team_a"

    )


with col2:

    team_b = st.selectbox(

        "Team B",

        teams,

        index=1 if len(teams) > 1 else 0,

        key="team_b"

    )


# ---------------------------------------
# TEAM A DATA
# ---------------------------------------

team_a_data = df[
    df[team_col] == team_a
]


# ---------------------------------------
# TEAM B DATA
# ---------------------------------------

team_b_data = df[
    df[team_col] == team_b
]


# ---------------------------------------
# LIVE INPUT
# ---------------------------------------

st.divider()

st.header("🎮 Live Match Input")


col1, col2 = st.columns(2)


# ---------------------------------------
# TEAM A INPUT
# ---------------------------------------

with col1:

    st.subheader(team_a)

    team_a_points = st.number_input(

        "Current Points",

        min_value=0,

        value=25,

        step=1,

        key="a_points"

    )


    team_a_kills = st.number_input(

        "Current Kills",

        min_value=0,

        value=8,

        step=1,

        key="a_kills"

    )


# ---------------------------------------
# TEAM B INPUT
# ---------------------------------------

with col2:

    st.subheader(team_b)

    team_b_points = st.number_input(

        "Current Points",

        min_value=0,

        value=15,

        step=1,

        key="b_points"

    )


    team_b_kills = st.number_input(

        "Current Kills",

        min_value=0,

        value=4,

        step=1,

        key="b_kills"

    )


# ---------------------------------------
# CALCULATE PROBABILITY
# ---------------------------------------

team_a_score = (

    team_a_points * 0.6

    +

    team_a_kills * 0.4

)


team_b_score = (

    team_b_points * 0.6

    +

    team_b_kills * 0.4

)


total_score = (

    team_a_score

    +

    team_b_score

)


if total_score == 0:

    team_a_probability = 50

    team_b_probability = 50


else:

    team_a_probability = (

        team_a_score

        /

        total_score

    ) * 100


    team_b_probability = (

        team_b_score

        /

        total_score

    ) * 100


# ---------------------------------------
# LIVE PROBABILITY
# ---------------------------------------

st.divider()

st.header("📊 Live Win Probability")


col1, col2 = st.columns(2)


# ---------------------------------------
# TEAM A RESULT
# ---------------------------------------

with col1:

    st.subheader(team_a)


    st.markdown(

        f"""
        <h1 style="text-align:center;">
        {team_a_probability:.2f}%
        </h1>
        """,

        unsafe_allow_html=True

    )


    st.progress(

        int(team_a_probability)

    )


    st.write(
        "Current Points"
    )

    st.metric(

        "Points",

        team_a_points

    )


    st.write(
        "Current Kills"
    )

    st.metric(

        "Kills",

        team_a_kills

    )


# ---------------------------------------
# TEAM B RESULT
# ---------------------------------------

with col2:

    st.subheader(team_b)


    st.markdown(

        f"""
        <h1 style="text-align:center;">
        {team_b_probability:.2f}%
        </h1>
        """,

        unsafe_allow_html=True

    )


    st.progress(

        int(team_b_probability)

    )


    st.write(
        "Current Points"
    )

    st.metric(

        "Points",

        team_b_points

    )


    st.write(
        "Current Kills"
    )

    st.metric(

        "Kills",

        team_b_kills

    )


# ---------------------------------------
# PREDICTED WINNER
# ---------------------------------------

st.divider()


if team_a_probability > team_b_probability:

    winner = team_a

    probability = team_a_probability


elif team_b_probability > team_a_probability:

    winner = team_b

    probability = team_b_probability


else:

    winner = "DRAW"

    probability = 50


# ---------------------------------------
# WINNER BOX
# ---------------------------------------

st.success(

    f"🏆 Live Predicted Winner: "
    f"{winner} "
    f"({probability:.2f}%)"

)


# ---------------------------------------
# SCORE COMPARISON
# ---------------------------------------

st.divider()

st.header("📈 Live Performance Comparison")


comparison_data = pd.DataFrame({

    "Metric": [

        "Points",

        "Kills",

        "Performance Score"

    ],

    team_a: [

        team_a_points,

        team_a_kills,

        round(team_a_score, 2)

    ],

    team_b: [

        team_b_points,

        team_b_kills,

        round(team_b_score, 2)

    ]

})


st.dataframe(

    comparison_data,

    use_container_width=True

)


# ---------------------------------------
# CHART DATA
# ---------------------------------------

chart_data = comparison_data.set_index(

    "Metric"

)


st.bar_chart(

    chart_data

)


# ---------------------------------------
# FOOTER
# ---------------------------------------

st.divider()

st.caption(

    "BGIS Win Probability System | Live Match Engine"

)