import streamlit as st


# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="About Project",
    page_icon="ℹ️",
    layout="wide"
)


# ---------------------------------------
# TITLE
# ---------------------------------------

st.title("ℹ️ About Project")

st.write(
    "BGIS Esports Analytics and Win Probability Prediction System"
)

st.divider()


# ---------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------

st.header("🎮 Project Overview")

st.write(
    """
    This project is an Esports Analytics and Win Probability
    Prediction System developed for BGIS tournament data.

    The system analyzes team performance using historical match
    data and predicts the probability of a team winning a match.
    """
)


# ---------------------------------------
# MAIN FEATURES
# ---------------------------------------

st.divider()

st.header("✨ Main Features")


col1, col2 = st.columns(2)


with col1:

    st.subheader("📊 Analytics")

    st.write(
        """
        • Tournament Dashboard

        • Team Analytics

        • Match History

        • Recent Form Analysis

        • Leaderboard
        """
    )


with col2:

    st.subheader("🤖 Machine Learning")

    st.write(
        """
        • Win Probability Prediction

        • Logistic Regression Model

        • Team Performance Analysis

        • Model Evaluation

        • Live Match Prediction
        """
    )


# ---------------------------------------
# PROJECT WORKFLOW
# ---------------------------------------

st.divider()

st.header("🔄 Project Workflow")


workflow_data = {

    "Step": [

        "1",

        "2",

        "3",

        "4",

        "5"

    ],

    "Process": [

        "Collect BGIS Data",

        "Clean and Prepare Data",

        "Feature Engineering",

        "Train Machine Learning Model",

        "Predict Win Probability"

    ]

}


st.dataframe(
    workflow_data,
    use_container_width=True
)


# ---------------------------------------
# TECHNOLOGIES
# ---------------------------------------

st.divider()

st.header("💻 Technologies Used")


tech_col1, tech_col2, tech_col3 = st.columns(3)


with tech_col1:

    st.subheader("🐍 Python")

    st.write(
        "Main programming language"
    )


with tech_col2:

    st.subheader("📊 Pandas")

    st.write(
        "Data analysis and processing"
    )


with tech_col3:

    st.subheader("🤖 Scikit-learn")

    st.write(
        "Machine learning model"
    )


# ---------------------------------------
# STREAMLIT
# ---------------------------------------

st.divider()

st.header("🌐 Application Framework")

st.info(
    "Streamlit is used to build the interactive "
    "web application and dashboard."
)


# ---------------------------------------
# PROJECT OBJECTIVE
# ---------------------------------------

st.divider()

st.header("🎯 Project Objective")

st.success(
    """
    The main objective of this project is to analyze BGIS
    esports team performance and estimate the probability
    of winning using machine learning techniques.
    """
)


# ---------------------------------------
# PROJECT PAGES
# ---------------------------------------

st.divider()

st.header("📱 Application Pages")


pages = [

    "🏠 Dashboard",

    "🔮 Match Predictor",

    "🔴 Live Match",

    "👥 Team Analytics",

    "🏆 Leaderboard",

    "📊 Tournament Statistics",

    "📜 Match History",

    "🤖 Model Evaluation",

    "ℹ️ About Project"

]


for page in pages:

    st.write(page)


# ---------------------------------------
# FOOTER
# ---------------------------------------

st.divider()

st.caption(
    "🎮 BGIS Win Probability System | "
    "Esports Analytics Project"
)