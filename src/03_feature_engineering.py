import streamlit as st
import pandas as pd
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Team Analytics",
    page_icon="👥",
    layout="wide"
)


# -----------------------------------
# LOAD DATA
# -----------------------------------

@st.cache_data
def load_data():

    possible_files = [
        "data/bgis_features.csv",
        "data/bgis_combined.csv",
        "data/features.csv",
        "data/processed_data.csv",
        "data/bgis_2026_updated.csv",
        "data/bgis_2026.csv",
        "data/bgis.csv",
        "data/bgis_data.csv",
        "data/matches.csv"
    ]

    for file in possible_files:

        if os.path.exists(file):

            df = pd.read_csv(file)

            if not df.empty:
                return df, file

    return None, None


df, loaded_file = load_data()


# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.title("👥 Team Analytics")

st.write(
    "Analyze team performance, points, kills, wins and tournament statistics."
)


# -----------------------------------
# CHECK DATA
# -----------------------------------

if df is None:

    st.warning("⚠️ Data file not found.")

    st.stop()


st.caption(f"📁 Data Source: {loaded_file}")


# -----------------------------------
# FIND TEAM COLUMN
# -----------------------------------

team_columns = [
    "team",
    "Team",
    "team_name",
    "Team Name"
]


team_col = None


for column in team_columns:

    if column in df.columns:

        team_col = column
        break


if team_col is None:

    st.error("❌ Team column not found.")

    st.write("Available columns:")

    st.write(df.columns.tolist())

    st.stop()


# -----------------------------------
# SIDEBAR FILTERS
# -----------------------------------

st.sidebar.header("🔍 Filters")


# Tournament Filter

if "tournament" in df.columns:

    tournaments = sorted(
        df["tournament"].dropna().unique()
    )

    selected_tournament = st.sidebar.selectbox(
        "Tournament",
        ["All"] + list(tournaments)
    )

    if selected_tournament != "All":

        df = df[
            df["tournament"] == selected_tournament
        ]


# Year Filter

if "year" in df.columns:

    years = sorted(
        df["year"].dropna().unique()
    )

    selected_year = st.sidebar.selectbox(
        "Year",
        ["All"] + list(years)
    )

    if selected_year != "All":

        df = df[
            df["year"] == selected_year
        ]


# Season Filter

if "season" in df.columns:

    seasons = sorted(
        df["season"].dropna().astype(str).unique()
    )

    selected_season = st.sidebar.selectbox(
        "Season",
        ["All"] + list(seasons)
    )

    if selected_season != "All":

        df = df[
            df["season"].astype(str)
            == str(selected_season)
        ]


# -----------------------------------
# SELECT TEAM
# -----------------------------------

teams = sorted(
    df[team_col].dropna().unique()
)


if len(teams) == 0:

    st.warning("No teams available for selected filters.")

    st.stop()


selected_team = st.selectbox(
    "Select Team",
    teams
)


# -----------------------------------
# TEAM DATA
# -----------------------------------

team_data = df[
    df[team_col] == selected_team
].copy()


# -----------------------------------
# TEAM TITLE
# -----------------------------------

st.header(f"📊 {selected_team}")


# -----------------------------------
# FIND COLUMNS
# -----------------------------------

def get_column(names):

    for name in names:

        if name in team_data.columns:

            return name

    return None


points_col = get_column([
    "points",
    "Points",
    "total_points",
    "Total Points"
])


kills_col = get_column([
    "kills",
    "Kills",
    "total_kills",
    "Total Kills"
])


win_col = get_column([
    "win",
    "Win",
    "wins",
    "Wins",
    "result",
    "Result"
])


elo_col = get_column([
    "elo_rating",
    "Elo Rating",
    "elo",
    "Elo"
])


# -----------------------------------
# CALCULATIONS
# -----------------------------------

total_matches = len(team_data)


# Total Points

total_points = 0

if points_col:

    total_points = pd.to_numeric(
        team_data[points_col],
        errors="coerce"
    ).fillna(0).sum()


# Total Kills

total_kills = 0

if kills_col:

    total_kills = pd.to_numeric(
        team_data[kills_col],
        errors="coerce"
    ).fillna(0).sum()


# Wins

wins = 0

if win_col:

    if team_data[win_col].dtype == object:

        wins = team_data[
            team_data[win_col]
            .astype(str)
            .str.upper()
            .isin(["WIN", "1", "TRUE"])
        ].shape[0]

    else:

        wins = pd.to_numeric(
            team_data[win_col],
            errors="coerce"
        ).fillna(0).sum()


# -----------------------------------
# ELO CALCULATION
# -----------------------------------

if elo_col:

    elo_rating = pd.to_numeric(
        team_data[elo_col],
        errors="coerce"
    ).mean()

    if pd.isna(elo_rating):

        elo_rating = 0


else:

    # Temporary calculated rating

    elo_rating = 700 + (
        total_points * 0.5
        + total_kills * 0.3
        + wins * 20
    )


# -----------------------------------
# WIN RATE
# -----------------------------------

win_rate = 0


if total_matches > 0:

    win_rate = (
        wins / total_matches
    ) * 100


# -----------------------------------
# AVERAGES
# -----------------------------------

average_points = 0


if total_matches > 0:

    average_points = (
        total_points / total_matches
    )


average_kills = 0


if total_matches > 0:

    average_kills = (
        total_kills / total_matches
    )


# -----------------------------------
# MAIN METRICS
# -----------------------------------

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Elo Rating",
        f"{elo_rating:.2f}"
    )


with col2:

    st.metric(
        "Matches",
        total_matches
    )


with col3:

    st.metric(
        "Total Points",
        int(total_points)
    )


with col4:

    st.metric(
        "Total Kills",
        int(total_kills)
    )


with col5:

    st.metric(
        "Wins",
        int(wins)
    )


# -----------------------------------
# TEAM SUMMARY
# -----------------------------------

st.divider()

st.header("📈 Performance Summary")


summary_col1, summary_col2, summary_col3 = st.columns(3)


with summary_col1:

    st.metric(
        "Average Points",
        f"{average_points:.2f}"
    )


with summary_col2:

    st.metric(
        "Average Kills",
        f"{average_kills:.2f}"
    )


with summary_col3:

    st.metric(
        "Win Rate",
        f"{win_rate:.2f}%"
    )


# -----------------------------------
# TEAM PERFORMANCE CHART
# -----------------------------------

st.divider()

st.header("📊 Team Performance")


if points_col and kills_col:

    chart_data = team_data[
        [points_col, kills_col]
    ].copy()


    chart_data.columns = [
        "Points",
        "Kills"
    ]


    st.line_chart(
        chart_data,
        use_container_width=True
    )


else:

    st.info(
        "Performance chart requires "
        "points and kills columns."
    )


# -----------------------------------
# POINTS CHART
# -----------------------------------

if points_col:

    st.subheader("🏆 Points Performance")

    points_data = team_data[
        [points_col]
    ].copy()

    points_data.columns = ["Points"]

    st.bar_chart(
        points_data,
        use_container_width=True
    )


# -----------------------------------
# KILLS CHART
# -----------------------------------

if kills_col:

    st.subheader("💀 Kills Performance")

    kills_data = team_data[
        [kills_col]
    ].copy()

    kills_data.columns = ["Kills"]

    st.bar_chart(
        kills_data,
        use_container_width=True
    )


# -----------------------------------
# RECENT MATCHES
# -----------------------------------

st.divider()

st.header("🔥 Recent Matches")


display_columns = []


if "match_id" in team_data.columns:

    display_columns.append(
        "match_id"
    )


if "tournament" in team_data.columns:

    display_columns.append(
        "tournament"
    )


if "year" in team_data.columns:

    display_columns.append(
        "year"
    )


if points_col:

    display_columns.append(
        points_col
    )


if kills_col:

    display_columns.append(
        kills_col
    )


if win_col:

    display_columns.append(
        win_col
    )


if len(display_columns) > 0:

    recent_matches = team_data[
        display_columns
    ].tail(10)


    st.dataframe(
        recent_matches,
        use_container_width=True,
        hide_index=True
    )


else:

    st.dataframe(
        team_data.tail(10),
        use_container_width=True,
        hide_index=True
    )


# -----------------------------------
# TOURNAMENT INFORMATION
# -----------------------------------

if "tournament" in team_data.columns:

    st.divider()

    st.header("🏆 Tournament Information")


    tournament_columns = []


    for column in [
        "tournament",
        "season",
        "year",
        "stage"
    ]:

        if column in team_data.columns:

            tournament_columns.append(
                column
            )


    if len(tournament_columns) > 0:

        tournament_data = team_data[
            tournament_columns
        ].drop_duplicates()


        st.dataframe(
            tournament_data,
            use_container_width=True,
            hide_index=True
        )


# -----------------------------------
# RAW TEAM DATA
# -----------------------------------

with st.expander("🔎 View Full Team Data"):

    st.dataframe(
        team_data,
        use_container_width=True,
        hide_index=True
    )


# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()


st.caption(
    "BGIS Esports Analytics | "
    "Tournament Performance & Win Probability Platform"
)