# import streamlit as st
# import pandas as pd
# import plotly.express as px

# from pathlib import Path


# # ============================================================
# # CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="Titanic Survival Analytics",
#     page_icon="🚢",
#     layout="wide"
# )


# # ============================================================
# # CUSTOM CSS
# # ============================================================

# st.markdown(
#     """
#     <style>

#     .main {
#         background-color: #F8FAFC;
#     }

#     h1 {
#         color: #0B1F3A;
#         font-weight: 800;
#     }

#     h2 {
#         color: #0B1F3A;
#     }

#     .metric-card {
#         background-color: white;
#         padding: 20px;
#         border-radius: 15px;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.06);
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # ============================================================
# # LOAD DATA
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parents[1]

# df = pd.read_csv(
#     BASE_DIR / "data/raw/train.csv"
# )


# # ============================================================
# # TITLE
# # ============================================================

# st.title(
#     "🚢 Titanic Survival Analytics"
# )

# st.markdown(
#     """
#     ### End-to-End Machine Learning & Exploratory Analysis

#     This dashboard analyzes which passenger characteristics
#     were associated with survival during the Titanic disaster.
#     """
# )


# # ============================================================
# # KPIs
# # ============================================================

# total_passengers = len(df)

# survivors = df["Survived"].sum()

# survival_rate = df["Survived"].mean() * 100

# female_survival = (
#     df.loc[df["Sex"] == "female", "Survived"]
#     .mean() * 100
# )

# male_survival = (
#     df.loc[df["Sex"] == "male", "Survived"]
#     .mean() * 100
# )


# col1, col2, col3, col4 = st.columns(4)

# col1.metric(
#     "Passengers",
#     f"{total_passengers:,}"
# )

# col2.metric(
#     "Survivors",
#     f"{survivors:,}"
# )

# col3.metric(
#     "Overall Survival",
#     f"{survival_rate:.1f}%"
# )

# col4.metric(
#     "Female Survival",
#     f"{female_survival:.1f}%"
# )


# # ============================================================
# # SIDEBAR
# # ============================================================

# st.sidebar.header("Filters")

# selected_sex = st.sidebar.multiselect(
#     "Gender",
#     options=df["Sex"].unique(),
#     default=df["Sex"].unique()
# )

# selected_class = st.sidebar.multiselect(
#     "Passenger Class",
#     options=sorted(df["Pclass"].unique()),
#     default=sorted(df["Pclass"].unique())
# )

# filtered = df[
#     df["Sex"].isin(selected_sex) &
#     df["Pclass"].isin(selected_class)
# ]


# # ============================================================
# # ROW 1
# # ============================================================

# col1, col2 = st.columns(2)


# with col1:

#     survival_gender = (
#         filtered.groupby("Sex")["Survived"]
#         .mean()
#         .reset_index()
#     )

#     survival_gender["Survival Rate"] = (
#         survival_gender["Survived"] * 100
#     )

#     fig = px.bar(
#         survival_gender,
#         x="Sex",
#         y="Survival Rate",
#         color="Sex",
#         title="Survival Rate by Gender",
#         color_discrete_map={
#             "female": "#7C3AED",
#             "male": "#2563EB"
#         },
#         text_auto=".1f"
#     )

#     fig.update_layout(
#         yaxis_title="Survival Rate (%)",
#         xaxis_title="Gender"
#     )

#     st.plotly_chart(
#         fig,
#         use_container_width=True
#     )


# with col2:

#     survival_class = (
#         filtered.groupby("Pclass")["Survived"]
#         .mean()
#         .reset_index()
#     )

#     survival_class["Survival Rate"] = (
#         survival_class["Survived"] * 100
#     )

#     fig = px.bar(
#         survival_class,
#         x="Pclass",
#         y="Survival Rate",
#         color="Pclass",
#         title="Survival Rate by Passenger Class",
#         color_continuous_scale="Blues",
#         text_auto=".1f"
#     )

#     fig.update_layout(
#         yaxis_title="Survival Rate (%)",
#         xaxis_title="Passenger Class"
#     )

#     st.plotly_chart(
#         fig,
#         use_container_width=True
#     )


# # ============================================================
# # ROW 2
# # ============================================================

# col1, col2 = st.columns(2)


# with col1:

#     fig = px.histogram(
#         filtered,
#         x="Age",
#         color="Survived",
#         nbins=30,
#         marginal="box",
#         title="Age Distribution by Survival",
#         color_discrete_map={
#             0: "#EF4444",
#             1: "#10B981"
#         }
#     )

#     st.plotly_chart(
#         fig,
#         use_container_width=True
#     )


# with col2:

#     fig = px.box(
#         filtered,
#         x="Pclass",
#         y="Fare",
#         color="Survived",
#         title="Fare Distribution by Class and Survival",
#         color_discrete_map={
#             0: "#EF4444",
#             1: "#10B981"
#         }
#     )

#     st.plotly_chart(
#         fig,
#         use_container_width=True
#     )


# # ============================================================
# # ROW 3
# # ============================================================

# filtered["FamilySize"] = (
#     filtered["SibSp"] +
#     filtered["Parch"] +
#     1
# )

# family = (
#     filtered.groupby("FamilySize")["Survived"]
#     .mean()
#     .reset_index()
# )

# family["Survival Rate"] = (
#     family["Survived"] * 100
# )

# fig = px.line(
#     family,
#     x="FamilySize",
#     y="Survival Rate",
#     markers=True,
#     title="Survival Rate by Family Size"
# )

# fig.update_traces(
#     line_color="#2563EB"
# )

# st.plotly_chart(
#     fig,
#     use_container_width=True
# )


# # ============================================================
# # DATA
# # ============================================================

# with st.expander("View Filtered Dataset"):

#     st.dataframe(
#         filtered,
#         use_container_width=True
#     )































































import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_FILE = (
    BASE_DIR
    / "outputs"
    / "models"
    / "titanic_survival_model.joblib"
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

# Make sure the project root is available for imports.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.feature_engineering import prepare_features


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Titanic Survival Analytics",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #0B1F3A;
        font-weight: 800;
    }

    h2 {
        color: #0B1F3A;
        font-weight: 700;
    }

    h3 {
        color: #0B1F3A;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }

    .dashboard-description {
        color: #475569;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #ECFDF5;
        border: 1px solid #10B981;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

@st.cache_data
def load_data():
    """
    Load Titanic training data.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    """
    Load the trained Titanic survival model.
    """

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


# ============================================================
# LOAD DATA
# ============================================================

try:
    df = load_data()

except Exception as exc:
    st.error(
        f"Unable to load the Titanic dataset.\n\n{exc}"
    )
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🚢 Titanic Survival Analytics"
)

st.markdown(
    """
    <div class="dashboard-description">

    ### End-to-End Machine Learning & Exploratory Analysis

    This interactive dashboard explores the factors associated
    with passenger survival during the Titanic disaster.

    Use the filters in the sidebar to investigate survival
    patterns by gender and passenger class.

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ Dashboard Controls")

st.sidebar.markdown(
    "Use the filters below to explore the dataset."
)

# Gender filter
sex_options = sorted(
    df["Sex"].dropna().unique().tolist()
)

selected_sex = st.sidebar.multiselect(
    "Gender",
    options=sex_options,
    default=sex_options
)

# Class filter
class_options = sorted(
    df["Pclass"].dropna().unique().tolist()
)

selected_class = st.sidebar.multiselect(
    "Passenger Class",
    options=class_options,
    default=class_options
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Dataset**

    Kaggle Titanic Training Dataset

    **Rows:** 891

    **Features:** 12

    **Target:** Survived
    """
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df[
    df["Sex"].isin(selected_sex)
    & df["Pclass"].isin(selected_class)
].copy()


# ============================================================
# EMPTY FILTER HANDLING
# ============================================================

if filtered.empty:

    st.warning(
        "No passengers match the selected filters. "
        "Please select at least one gender and one passenger class."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_passengers = len(filtered)

survivors = int(
    filtered["Survived"].sum()
)

survival_rate = (
    filtered["Survived"].mean() * 100
)

female_data = filtered.loc[
    filtered["Sex"] == "female",
    "Survived"
]

male_data = filtered.loc[
    filtered["Sex"] == "male",
    "Survived"
]

female_survival = (
    female_data.mean() * 100
    if not female_data.empty
    else 0
)

male_survival = (
    male_data.mean() * 100
    if not male_data.empty
    else 0
)


# ============================================================
# KPI SECTION
# ============================================================

st.header("📊 Key Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Passengers",
    f"{total_passengers:,}"
)

col2.metric(
    "Survivors",
    f"{survivors:,}"
)

col3.metric(
    "Overall Survival",
    f"{survival_rate:.1f}%"
)

col4.metric(
    "Female Survival",
    f"{female_survival:.1f}%"
)

col5.metric(
    "Male Survival",
    f"{male_survival:.1f}%"
)


# ============================================================
# ROW 1
# ============================================================

st.header("👥 Survival Analysis")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# SURVIVAL BY GENDER
# ------------------------------------------------------------

with col1:

    survival_gender = (
        filtered
        .groupby("Sex", as_index=False)["Survived"]
        .mean()
    )

    survival_gender["Survival Rate"] = (
        survival_gender["Survived"] * 100
    )

    fig = px.bar(
        survival_gender,
        x="Sex",
        y="Survival Rate",
        color="Sex",
        title="Survival Rate by Gender",
        color_discrete_map={
            "female": "#7C3AED",
            "male": "#2563EB"
        },
        text="Survival Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Survival Rate (%)",
        xaxis_title="Gender",
        yaxis_range=[0, 100],
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------------------
# SURVIVAL BY CLASS
# ------------------------------------------------------------

with col2:

    survival_class = (
        filtered
        .groupby("Pclass", as_index=False)["Survived"]
        .mean()
    )

    survival_class["Survival Rate"] = (
        survival_class["Survived"] * 100
    )

    fig = px.bar(
        survival_class,
        x="Pclass",
        y="Survival Rate",
        color="Pclass",
        title="Survival Rate by Passenger Class",
        color_continuous_scale="Blues",
        text="Survival Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Survival Rate (%)",
        xaxis_title="Passenger Class",
        yaxis_range=[0, 100],
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# ROW 2
# ============================================================

st.header("📈 Passenger Characteristics")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# AGE DISTRIBUTION
# ------------------------------------------------------------

with col1:

    fig = px.histogram(
        filtered,
        x="Age",
        color="Survived",
        nbins=30,
        marginal="box",
        title="Age Distribution by Survival",
        color_discrete_map={
            0: "#EF4444",
            1: "#10B981"
        },
        labels={
            "Survived": "Survival Status"
        }
    )

    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Passengers",
        legend_title="Survival"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------------------
# FARE DISTRIBUTION
# ------------------------------------------------------------

with col2:

    fig = px.box(
        filtered,
        x="Pclass",
        y="Fare",
        color="Survived",
        title="Fare Distribution by Class and Survival",
        color_discrete_map={
            0: "#EF4444",
            1: "#10B981"
        },
        labels={
            "Pclass": "Passenger Class",
            "Fare": "Fare",
            "Survived": "Survival Status"
        }
    )

    fig.update_layout(
        yaxis_title="Fare",
        xaxis_title="Passenger Class"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# ROW 3 - FAMILY SIZE
# ============================================================

st.header("👨‍👩‍👧 Family Analysis")

family_data = filtered.copy()

family_data["FamilySize"] = (
    family_data["SibSp"]
    + family_data["Parch"]
    + 1
)

family = (
    family_data
    .groupby("FamilySize", as_index=False)["Survived"]
    .mean()
)

family["Survival Rate"] = (
    family["Survived"] * 100
)

fig = px.line(
    family,
    x="FamilySize",
    y="Survival Rate",
    markers=True,
    title="Survival Rate by Family Size"
)

fig.update_traces(
    line_color="#2563EB",
    marker_color="#2563EB"
)

fig.update_layout(
    xaxis_title="Family Size",
    yaxis_title="Survival Rate (%)",
    yaxis_range=[0, 100]
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ROW 4 - GENDER + CLASS
# ============================================================

st.header("🎯 Gender and Class Interaction")

gender_class = (
    filtered
    .groupby(
        ["Pclass", "Sex"],
        as_index=False
    )["Survived"]
    .mean()
)

gender_class["Survival Rate"] = (
    gender_class["Survived"] * 100
)

fig = px.bar(
    gender_class,
    x="Pclass",
    y="Survival Rate",
    color="Sex",
    barmode="group",
    title="Survival Rate by Gender and Passenger Class",
    color_discrete_map={
        "female": "#7C3AED",
        "male": "#2563EB"
    },
    text="Survival Rate"
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Passenger Class",
    yaxis_title="Survival Rate (%)",
    yaxis_range=[0, 100]
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# MACHINE LEARNING SECTION
# ============================================================

st.header("🤖 Machine Learning Model")

st.markdown(
    """
    The final machine learning pipeline uses **Gradient Boosting**
    selected using cross-validation based on ROC-AUC.
    """
)

model_col1, model_col2, model_col3 = st.columns(3)

model_col1.metric(
    "Model",
    "Gradient Boosting"
)

model_col2.metric(
    "Cross-Validation ROC-AUC",
    "89.17%"
)

model_col3.metric(
    "Cross-Validation Accuracy",
    "82.71%"
)


# ============================================================
# SINGLE PASSENGER PREDICTION
# ============================================================

st.subheader("🔮 Predict Passenger Survival")

st.write(
    "Enter passenger information below to obtain a model prediction."
)

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        passenger_class = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            index=2
        )

        sex = st.selectbox(
            "Gender",
            options=["male", "female"]
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0
        )

    with col2:

        sib_sp = st.number_input(
            "Siblings / Spouses",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        parch = st.number_input(
            "Parents / Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        fare = st.number_input(
            "Fare",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=1.0
        )

    with col3:

        embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            index=0
        )

        ticket = st.text_input(
            "Ticket",
            value="A/5 21171"
        )

        cabin = st.text_input(
            "Cabin",
            value=""
        )

        name = st.text_input(
            "Name",
            value="Passenger, Mr."
        )

    submitted = st.form_submit_button(
        "🚢 Predict Survival"
    )


# ============================================================
# RUN PREDICTION
# ============================================================

if submitted:

    try:

        model = load_model()

        passenger = pd.DataFrame(
            [{
                "PassengerId": 99999,
                "Pclass": passenger_class,
                "Name": name,
                "Sex": sex,
                "Age": age,
                "SibSp": sib_sp,
                "Parch": parch,
                "Ticket": ticket,
                "Fare": fare,
                "Cabin": cabin if cabin else pd.NA,
                "Embarked": embarked
            }]
        )

        features = prepare_features(
            passenger
        )

        prediction = int(
            model.predict(features)[0]
        )

        probability = float(
            model.predict_proba(features)[0, 1]
        )

        st.markdown("---")

        if prediction == 1:

            st.success(
                f"### 🟢 Predicted: Survived\n\n"
                f"Survival probability: **{probability:.1%}**"
            )

        else:

            st.error(
                f"### 🔴 Predicted: Did Not Survive\n\n"
                f"Survival probability: **{probability:.1%}**"
            )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )


# ============================================================
# FILTERED DATASET
# ============================================================

st.header("📋 Filtered Dataset")

with st.expander(
    "View Filtered Dataset"
):

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA SUMMARY
# ============================================================

st.header("📌 Dataset Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:

    st.write("**Dataset Shape**")

    st.write(
        f"Rows: {df.shape[0]:,}"
    )

    st.write(
        f"Columns: {df.shape[1]:,}"
    )

with summary_col2:

    st.write("**Missing Values**")

    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:

        st.success(
            "No missing values found."
        )

    else:

        st.dataframe(
            missing.rename(
                "Missing Values"
            ),
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Titanic Survival Classification | "
    "Data Science Project | "
    "Machine Learning + EDA + Interactive Analytics"
)








# for run the code |--> (run it in powershell in the same folder)

# python -m streamlit run app/app.py
