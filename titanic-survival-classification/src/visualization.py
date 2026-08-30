# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# from pathlib import Path

# from config import FIGURE_DIR, COLORS


# # ============================================================
# # GLOBAL STYLE
# # ============================================================

# sns.set_theme(
#     style="whitegrid",
#     font_scale=1.05
# )

# plt.rcParams["figure.facecolor"] = COLORS["white"]
# plt.rcParams["axes.facecolor"] = COLORS["white"]
# plt.rcParams["axes.edgecolor"] = COLORS["light_gray"]
# plt.rcParams["axes.titleweight"] = "bold"


# def save_figure(filename):
#     path = FIGURE_DIR / filename

#     plt.tight_layout()
#     plt.savefig(
#         path,
#         dpi=300,
#         bbox_inches="tight",
#         facecolor="white"
#     )

#     plt.close()

#     print(f"Saved: {path}")


# # ============================================================
# # 1. TARGET DISTRIBUTION
# # ============================================================

# def plot_survival_distribution(df):

#     plt.figure(figsize=(9, 6))

#     ax = sns.countplot(
#         data=df,
#         x="Survived",
#         hue="Survived",
#         palette=[
#             COLORS["red"],
#             COLORS["green"]
#         ],
#         legend=False
#     )

#     plt.title(
#         "Titanic Survival Distribution",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Survival Status")
#     plt.ylabel("Number of Passengers")

#     ax.set_xticklabels([
#         "Did Not Survive",
#         "Survived"
#     ])

#     save_figure("01_survival_distribution.png")


# # ============================================================
# # 2. GENDER VS SURVIVAL
# # ============================================================

# def plot_gender_survival(df):

#     plt.figure(figsize=(10, 6))

#     ax = sns.barplot(
#         data=df,
#         x="Sex",
#         y="Survived",
#         hue="Sex",
#         palette=[
#             COLORS["blue"],
#             COLORS["purple"]
#         ],
#         errorbar=None,
#         legend=False
#     )

#     plt.title(
#         "Survival Rate by Gender",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Gender")
#     plt.ylabel("Survival Rate")

#     plt.ylim(0, 1)

#     for container in ax.containers:
#         ax.bar_label(
#             container,
#             fmt="%.2f"
#         )

#     save_figure("02_gender_survival.png")


# # ============================================================
# # 3. CLASS VS SURVIVAL
# # ============================================================

# def plot_class_survival(df):

#     plt.figure(figsize=(10, 6))

#     ax = sns.barplot(
#         data=df,
#         x="Pclass",
#         y="Survived",
#         hue="Pclass",
#         palette=[
#             COLORS["gold"],
#             COLORS["blue"],
#             COLORS["red"]
#         ],
#         errorbar=None,
#         legend=False
#     )

#     plt.title(
#         "Survival Rate by Passenger Class",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Passenger Class")
#     plt.ylabel("Survival Rate")

#     plt.ylim(0, 1)

#     save_figure("03_class_survival.png")


# # ============================================================
# # 4. GENDER + CLASS
# # ============================================================

# def plot_gender_class_survival(df):

#     grouped = (
#         df.groupby(["Pclass", "Sex"])["Survived"]
#         .mean()
#         .reset_index()
#     )

#     plt.figure(figsize=(11, 7))

#     sns.barplot(
#         data=grouped,
#         x="Pclass",
#         y="Survived",
#         hue="Sex",
#         palette={
#             "male": COLORS["blue"],
#             "female": COLORS["purple"]
#         }
#     )

#     plt.title(
#         "Survival Rate by Passenger Class and Gender",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Passenger Class")
#     plt.ylabel("Survival Rate")

#     plt.ylim(0, 1)

#     save_figure("04_gender_class_survival.png")


# # ============================================================
# # 5. AGE DISTRIBUTION
# # ============================================================

# def plot_age_distribution(df):

#     plt.figure(figsize=(11, 6))

#     sns.histplot(
#         data=df,
#         x="Age",
#         hue="Survived",
#         bins=30,
#         kde=True,
#         palette={
#             0: COLORS["red"],
#             1: COLORS["green"]
#         },
#         alpha=0.45
#     )

#     plt.title(
#         "Age Distribution by Survival Status",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Age")
#     plt.ylabel("Passengers")

#     save_figure("05_age_distribution.png")


# # ============================================================
# # 6. AGE BOX PLOT
# # ============================================================

# def plot_age_boxplot(df):

#     plt.figure(figsize=(10, 6))

#     sns.boxplot(
#         data=df,
#         x="Survived",
#         y="Age",
#         hue="Survived",
#         palette={
#             0: COLORS["red"],
#             1: COLORS["green"]
#         },
#         legend=False
#     )

#     plt.title(
#         "Age Distribution by Survival",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Survival Status")
#     plt.ylabel("Age")

#     save_figure("06_age_boxplot.png")


# # ============================================================
# # 7. FARE
# # ============================================================

# def plot_fare_survival(df):

#     plt.figure(figsize=(11, 6))

#     sns.boxplot(
#         data=df,
#         x="Pclass",
#         y="Fare",
#         hue="Survived",
#         palette={
#             0: COLORS["red"],
#             1: COLORS["green"]
#         }
#     )

#     plt.title(
#         "Fare Distribution by Class and Survival",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Passenger Class")
#     plt.ylabel("Fare")

#     plt.ylim(
#         0,
#         df["Fare"].quantile(0.98)
#     )

#     save_figure("07_fare_survival.png")


# # ============================================================
# # 8. FAMILY SIZE
# # ============================================================

# def plot_family_size(df):

#     temp = df.copy()

#     temp["FamilySize"] = (
#         temp["SibSp"] +
#         temp["Parch"] +
#         1
#     )

#     grouped = (
#         temp.groupby("FamilySize")["Survived"]
#         .mean()
#         .reset_index()
#     )

#     plt.figure(figsize=(12, 6))

#     sns.barplot(
#         data=grouped,
#         x="FamilySize",
#         y="Survived",
#         color=COLORS["blue"]
#     )

#     plt.title(
#         "Survival Rate by Family Size",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Family Size")
#     plt.ylabel("Survival Rate")

#     plt.ylim(0, 1)

#     save_figure("08_family_size_survival.png")


# # ============================================================
# # 9. EMBARKED
# # ============================================================

# def plot_embarked_survival(df):

#     plt.figure(figsize=(10, 6))

#     sns.barplot(
#         data=df,
#         x="Embarked",
#         y="Survived",
#         hue="Embarked",
#         palette=[
#             COLORS["blue"],
#             COLORS["gold"],
#             COLORS["purple"]
#         ],
#         errorbar=None,
#         legend=False
#     )

#     plt.title(
#         "Survival Rate by Port of Embarkation",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Embarkation Port")
#     plt.ylabel("Survival Rate")

#     plt.ylim(0, 1)

#     save_figure("09_embarked_survival.png")


# # ============================================================
# # 10. CORRELATION
# # ============================================================

# def plot_correlation(df):

#     numeric = df.select_dtypes(
#         include=np.number
#     )

#     correlation = numeric.corr()

#     plt.figure(figsize=(12, 9))

#     sns.heatmap(
#         correlation,
#         annot=True,
#         fmt=".2f",
#         cmap="RdYlBu_r",
#         center=0,
#         linewidths=0.5
#     )

#     plt.title(
#         "Numerical Feature Correlation Matrix",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     save_figure("10_correlation_heatmap.png")


# # ============================================================
# # MASTER FUNCTION
# # ============================================================

# def generate_all_visualizations(df):

#     plot_survival_distribution(df)
#     plot_gender_survival(df)
#     plot_class_survival(df)
#     plot_gender_class_survival(df)
#     plot_age_distribution(df)
#     plot_age_boxplot(df)
#     plot_fare_survival(df)
#     plot_family_size(df)
#     plot_embarked_survival(df)
#     plot_correlation(df)

#     print("\nAll visualizations generated.")





































































# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from config import FIGURE_DIR, COLORS


# # ============================================================
# # GLOBAL STYLE
# # ============================================================

# sns.set_theme(
#     style="whitegrid",
#     font_scale=1.05
# )

# plt.rcParams["figure.facecolor"] = COLORS["white"]
# plt.rcParams["axes.facecolor"] = COLORS["white"]
# plt.rcParams["axes.edgecolor"] = COLORS["light_gray"]
# plt.rcParams["axes.titleweight"] = "bold"


# # ============================================================
# # HELPER FUNCTIONS
# # ============================================================

# def validate_columns(df, required_columns):
#     """
#     Validate that the DataFrame contains all required columns.

#     Args:
#         df (pd.DataFrame): Input DataFrame.
#         required_columns (set): Required column names.

#     Raises:
#         TypeError: If df is not a pandas DataFrame.
#         ValueError: If required columns are missing.
#     """

#     if not isinstance(df, pd.DataFrame):
#         raise TypeError(
#             "df must be a pandas DataFrame."
#         )

#     missing_columns = set(required_columns) - set(df.columns)

#     if missing_columns:
#         raise ValueError(
#             f"Missing required columns: "
#             f"{sorted(missing_columns)}"
#         )


# def save_figure(filename):
#     """
#     Save the current Matplotlib figure.

#     Args:
#         filename (str): Output filename.
#     """

#     # Make sure the output directory exists.
#     FIGURE_DIR.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     path = FIGURE_DIR / filename

#     plt.tight_layout()

#     plt.savefig(
#         path,
#         dpi=300,
#         bbox_inches="tight",
#         facecolor=COLORS["white"]
#     )

#     plt.close()

#     print(f"Saved: {path}")


# # ============================================================
# # 1. TARGET DISTRIBUTION
# # ============================================================

# def plot_survival_distribution(df):
#     """
#     Plot the distribution of survival outcomes.
#     """

#     validate_columns(
#         df,
#         {"Survived"}
#     )

#     plot_df = df.dropna(
#         subset=["Survived"]
#     ).copy()

#     plt.figure(figsize=(9, 6))

#     ax = sns.countplot(
#         data=plot_df,
#         x="Survived",
#         hue="Survived",
#         order=[0, 1],
#         hue_order=[0, 1],
#         palette={
#             0: COLORS["red"],
#             1: COLORS["green"]
#         },
#         legend=False
#     )

#     ax.set_title(
#         "Titanic Survival Distribution",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     ax.set_xlabel("Survival Status")
#     ax.set_ylabel("Number of Passengers")

#     ax.set_xticks([0, 1])
#     ax.set_xticklabels([
#         "Did Not Survive",
#         "Survived"
#     ])

#     save_figure(
#         "01_survival_distribution.png"
#     )


# # ============================================================
# # 2. GENDER VS SURVIVAL
# # ============================================================

# def plot_gender_survival(df):
#     """
#     Plot survival rate by gender.
#     """

#     validate_columns(
#         df,
#         {"Sex", "Survived"}
#     )

#     plot_df = df.dropna(
#         subset=["Sex", "Survived"]
#     ).copy()

#     plt.figure(figsize=(10, 6))

#     ax = sns.barplot(
#         data=plot_df,
#         x="Sex",
#         y="Survived",
#         hue="Sex",
#         order=["male", "female"],
#         palette={
#             "male": COLORS["blue"],
#             "female": COLORS["purple"]
#         },
#         errorbar=None,
#         legend=False
#     )

#     ax.set_title(
#         "Survival Rate by Gender",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     ax.set_xlabel("Gender")
#     ax.set_ylabel("Survival Rate")
#     ax.set_ylim(0, 1)

#     for container in ax.containers:
#         ax.bar_label(
#             container,
#             fmt="%.2f",
#             padding=3
#         )

#     save_figure(
#         "02_gender_survival.png"
#     )


# # ============================================================
# # 3. CLASS VS SURVIVAL
# # ============================================================

# def plot_class_survival(df):
#     """
#     Plot survival rate by passenger class.
#     """

#     validate_columns(
#         df,
#         {"Pclass", "Survived"}
#     )

#     plot_df = df.dropna(
#         subset=["Pclass", "Survived"]
#     ).copy()

#     plot_df["Pclass"] = plot_df["Pclass"].astype(int)

#     plt.figure(figsize=(10, 6))

#     ax = sns.barplot(
#         data=plot_df,
#         x="Pclass",
#         y="Survived",
#         hue="Pclass",
#         order=[1, 2, 3],
#         hue_order=[1, 2, 3],
#         palette={
#             1: COLORS["gold"],
#             2: COLORS["blue"],
#             3: COLORS["red"]
#         },
#         errorbar=None,
#         legend=False
#     )

#     ax.set_title(
#         "Survival Rate by Passenger Class",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     ax.set_xlabel("Passenger Class")
#     ax.set_ylabel("Survival Rate")
#     ax.set_ylim(0, 1)

#     for container in ax.containers:
#         ax.bar_label(
#             container,
#             fmt="%.2f",
#             padding=3
#         )

#     save_figure(
#         "03_class_survival.png"
#     )


# # ============================================================
# # 4. GENDER + CLASS
# # ============================================================

# def plot_gender_class_survival(df):
#     """
#     Plot survival rate by passenger class and gender.
#     """

#     validate_columns(
#         df,
#         {"Pclass", "Sex", "Survived"}
#     )

#     plot_df = df.dropna(
#         subset=["Pclass", "Sex", "Survived"]
#     ).copy()

#     grouped = (
#         plot_df
#         .groupby(
#             ["Pclass", "Sex"],
#             as_index=False
#         )["Survived"]
#         .mean()
#     )

#     plt.figure(figsize=(11, 7))

#     ax = sns.barplot(
#         data=grouped,
#         x="Pclass",
#         y="Survived",
#         hue="Sex",
#         order=[1, 2, 3],
#         hue_order=["male", "female"],
#         palette={
#             "male": COLORS["blue"],
#             "female": COLORS["purple"]
#         },
#         errorbar=None
#     )

#     ax.set_title(
#         "Survival Rate by Passenger Class and Gender",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     ax.set_xlabel("Passenger Class")
#     ax.set_ylabel("Survival Rate")
#     ax.set_ylim(0, 1)

#     for container in ax.containers:
#         ax.bar_label(
#             container,
#             fmt="%.2f",
#             padding=3
#         )

#     save_figure(
#         "04_gender_class_survival.png"
#     )


# # ============================================================
# # 5. AGE DISTRIBUTION
# # ============================================================

# def plot_age_distribution(df):
#     """
#     Plot age distribution by survival status.
#     """

#     validate_columns(
#         df,
#         {"Age", "Survived"}
#     )

#     plot_df = df.dropna(
#         subset=["Age", "Survived"]
#     ).copy()

#     plt.figure(figsize=(11, 6))

#     sns.histplot(
#         data=plot_df,
#         x="Age",
#         hue="Survived",
#         bins=30,
#         kde=True,
#         hue_order=[0, 1],
#         palette={
#             0: COLORS["red"],
#             1: COLORS["green"]
#         },
#         alpha=0.45
#     )

#     plt.title(
#         "Age Distribution by Survival Status",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     plt.xlabel("Age")
#     plt.ylabel("Passengers")

#     save_figure(
#         "05_age_distribution.png"
#     )


# # ============================================================
# # 6. AGE BOX PLOT
# # ============================================================

# def plot_age_boxplot(df):
#     """
#     Plot age distribution by survival status.
#     """

#     validate_columns(
#         df,
#         {"Age", "Survived"}
#     )

#     plot_df = df.dropna(
#         subset=["Age", "Survived"]
#     ).copy()

#     plt.figure(figsize=(10, 6))

#     ax = sns.boxplot(
#         data=plot_df,
#         x="Survived",
#         y="Age",
#         hue="Survived",
#         order=[0, 1],
#         hue_order=[0, 1],
#         palette={
#             0: COLORS["red"],
#             1: COLORS["green"]
#         },
#         legend=False
#     )

#     ax.set_title(
#         "Age Distribution by Survival",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     ax.set_xlabel("Survival Status")
#     ax.set_ylabel("Age")

#     ax.set_xticks([0, 1])
#     ax.set_xticklabels([
#         "Did Not Survive",
#         "Survived"
#     ])

#     save_figure(
#         "06_age_boxplot.png"
#     )


# # ============================================================
# # 7. FARE
# # ============================================================

# def plot_fare_survival(df):
#     """
#     Plot fare distribution by passenger class and survival.
#     """

#     validate_columns(
#         df,
#         {"Pclass", "Fare", "Survived"}
#     )

#     plot_df = df.dropna(
#         subset=["Pclass", "Fare", "Survived"]
#     ).copy()

#     plot_df["Pclass"] = plot_df["Pclass"].astype(int)

#     # Remove extreme fare values from the visible plot range.
#     fare_limit = plot_df["Fare"].quantile(0.98)

#     plt.figure(figsize=(11, 6))

#     ax = sns.boxplot(
#         data=plot_df,
#         x="Pclass",
#         y="Fare",
#         hue="Survived",
#         order=[1, 2, 3],
#         hue_order=[0, 1],
#         palette={
#             0: COLORS["red"],
#             1: COLORS["green"]
#         }
#     )

#     ax.set_title(
#         "Fare Distribution by Class and Survival",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     ax.set_xlabel("Passenger Class")
#     ax.set_ylabel("Fare")

#     ax.set_ylim(
#         0,
#         fare_limit
#     )

#     save_figure(
#         "07_fare_survival.png"
#     )


# # ============================================================
# # 8. FAMILY SIZE
# # ============================================================

# def plot_family_size(df):
#     """
#     Plot survival rate by family size.

#     FamilySize = SibSp + Parch + 1
#     """

#     validate_columns(
#         df,
#         {"SibSp", "Parch", "Survived"}
#     )

#     plot_df = df.dropna(
#         subset=["SibSp", "Parch", "Survived"]
#     ).copy()

#     plot_df["FamilySize"] = (
#         plot_df["SibSp"]
#         + plot_df["Parch"]
#         + 1
#     )

#     grouped = (
#         plot_df
#         .groupby(
#             "FamilySize",
#             as_index=False
#         )["Survived"]
#         .mean()
#     )

#     plt.figure(figsize=(12, 6))

#     ax = sns.barplot(
#         data=grouped,
#         x="FamilySize",
#         y="Survived",
#         color=COLORS["blue"],
#         errorbar=None
#     )

#     ax.set_title(
#         "Survival Rate by Family Size",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     ax.set_xlabel("Family Size")
#     ax.set_ylabel("Survival Rate")
#     ax.set_ylim(0, 1)

#     save_figure(
#         "08_family_size_survival.png"
#     )


# # ============================================================
# # 9. EMBARKED
# # ============================================================

# def plot_embarked_survival(df):
#     """
#     Plot survival rate by port of embarkation.

#     S = Southampton
#     C = Cherbourg
#     Q = Queenstown
#     """

#     validate_columns(
#         df,
#         {"Embarked", "Survived"}
#     )

#     plot_df = df.dropna(
#         subset=["Embarked", "Survived"]
#     ).copy()

#     embarked_order = ["S", "C", "Q"]

#     plt.figure(figsize=(10, 6))

#     ax = sns.barplot(
#         data=plot_df,
#         x="Embarked",
#         y="Survived",
#         hue="Embarked",
#         order=embarked_order,
#         hue_order=embarked_order,
#         palette={
#             "S": COLORS["blue"],
#             "C": COLORS["gold"],
#             "Q": COLORS["purple"]
#         },
#         errorbar=None,
#         legend=False
#     )

#     ax.set_title(
#         "Survival Rate by Port of Embarkation",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     ax.set_xlabel("Embarkation Port")
#     ax.set_ylabel("Survival Rate")
#     ax.set_ylim(0, 1)

#     ax.set_xticks([0, 1, 2])
#     ax.set_xticklabels([
#         "Southampton (S)",
#         "Cherbourg (C)",
#         "Queenstown (Q)"
#     ])

#     for container in ax.containers:
#         ax.bar_label(
#             container,
#             fmt="%.2f",
#             padding=3
#         )

#     save_figure(
#         "09_embarked_survival.png"
#     )


# # ============================================================
# # 10. CORRELATION
# # ============================================================

# def plot_correlation(df):
#     """
#     Plot the correlation matrix for useful numeric features.

#     PassengerId is excluded because it is an identifier,
#     not a meaningful predictive feature.
#     """

#     if not isinstance(df, pd.DataFrame):
#         raise TypeError(
#             "df must be a pandas DataFrame."
#         )

#     numeric = df.select_dtypes(
#         include=np.number
#     ).copy()

#     # PassengerId is an identifier and should not be included
#     # in feature correlation analysis.
#     numeric = numeric.drop(
#         columns=["PassengerId"],
#         errors="ignore"
#     )

#     if numeric.empty:
#         raise ValueError(
#             "No numeric columns available for correlation analysis."
#         )

#     correlation = numeric.corr()

#     plt.figure(figsize=(12, 9))

#     sns.heatmap(
#         correlation,
#         annot=True,
#         fmt=".2f",
#         cmap="RdYlBu_r",
#         center=0,
#         linewidths=0.5,
#         square=True
#     )

#     plt.title(
#         "Numerical Feature Correlation Matrix",
#         fontsize=18,
#         color=COLORS["navy"]
#     )

#     save_figure(
#         "10_correlation_heatmap.png"
#     )


# # ============================================================
# # MASTER FUNCTION
# # ============================================================

# def generate_all_visualizations(df):
#     """
#     Generate all Titanic exploratory visualizations.

#     Args:
#         df (pd.DataFrame): Titanic training dataset.
#     """

#     if not isinstance(df, pd.DataFrame):
#         raise TypeError(
#             "df must be a pandas DataFrame."
#         )

#     print("\nGenerating visualizations...")
#     print("-" * 70)

#     plot_survival_distribution(df)
#     plot_gender_survival(df)
#     plot_class_survival(df)
#     plot_gender_class_survival(df)
#     plot_age_distribution(df)
#     plot_age_boxplot(df)
#     plot_fare_survival(df)
#     plot_family_size(df)
#     plot_embarked_survival(df)
#     plot_correlation(df)

#     print("-" * 70)
#     print("All visualizations generated successfully.")


# # ============================================================
# # OPTIONAL DIRECT EXECUTION
# # ============================================================

# if __name__ == "__main__":
#     from data_loader import load_train_data

#     train = load_train_data()

#     generate_all_visualizations(train)

















































import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .config import FIGURE_DIR, COLORS


# ============================================================
# GLOBAL STYLE
# ============================================================

sns.set_theme(
    style="whitegrid",
    font_scale=1.05,
)

plt.rcParams["figure.facecolor"] = COLORS["white"]
plt.rcParams["axes.facecolor"] = COLORS["white"]
plt.rcParams["axes.edgecolor"] = COLORS["light_gray"]
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelcolor"] = COLORS["navy"]
plt.rcParams["xtick.color"] = COLORS["gray"]
plt.rcParams["ytick.color"] = COLORS["gray"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_columns(df, required_columns):
    """
    Validate that a DataFrame contains required columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (set): Required column names.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If required columns are missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    missing_columns = (
        set(required_columns) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )


def save_figure(filename):
    """
    Save the current Matplotlib figure.

    Args:
        filename (str): Output filename.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = FIGURE_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor=COLORS["white"],
    )

    plt.close()

    print(f"Saved: {path}")


def add_bar_labels(ax, decimals=2):
    """
    Add numeric labels to bar charts.

    Args:
        ax: Matplotlib Axes object.
        decimals (int): Number of decimal places.
    """

    for container in ax.containers:
        labels = []

        for bar in container:
            height = bar.get_height()

            if np.isfinite(height):
                labels.append(
                    f"{height:.{decimals}f}"
                )
            else:
                labels.append("")

        try:
            ax.bar_label(
                container,
                labels=labels,
                padding=3,
                fontsize=9,
            )
        except (AttributeError, TypeError):
            pass


# ============================================================
# 1. TARGET DISTRIBUTION
# ============================================================

def plot_survival_distribution(df):
    """
    Plot the distribution of survival outcomes.
    """

    validate_columns(
        df,
        {"Survived"},
    )

    plot_df = df.dropna(
        subset=["Survived"],
    ).copy()

    plot_df["Survived"] = (
        plot_df["Survived"].astype(int)
    )

    plt.figure(
        figsize=(9, 6)
    )

    ax = sns.countplot(
        data=plot_df,
        x="Survived",
        hue="Survived",
        order=[0, 1],
        hue_order=[0, 1],
        palette={
            0: COLORS["red"],
            1: COLORS["green"],
        },
        legend=False,
    )

    ax.set_title(
        "Titanic Survival Distribution",
        fontsize=18,
        color=COLORS["navy"],
    )

    ax.set_xlabel(
        "Survival Status"
    )

    ax.set_ylabel(
        "Number of Passengers"
    )

    ax.set_xticks([0, 1])

    ax.set_xticklabels(
        [
            "Did Not Survive",
            "Survived",
        ]
    )

    add_bar_labels(
        ax,
        decimals=0,
    )

    save_figure(
        "01_survival_distribution.png"
    )


# ============================================================
# 2. GENDER VS SURVIVAL
# ============================================================

def plot_gender_survival(df):
    """
    Plot survival rate by gender.
    """

    validate_columns(
        df,
        {"Sex", "Survived"},
    )

    plot_df = df.dropna(
        subset=["Sex", "Survived"],
    ).copy()

    plt.figure(
        figsize=(10, 6)
    )

    ax = sns.barplot(
        data=plot_df,
        x="Sex",
        y="Survived",
        hue="Sex",
        order=["male", "female"],
        palette={
            "male": COLORS["blue"],
            "female": COLORS["purple"],
        },
        errorbar=None,
        legend=False,
    )

    ax.set_title(
        "Survival Rate by Gender",
        fontsize=18,
        color=COLORS["navy"],
    )

    ax.set_xlabel("Gender")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1)

    add_bar_labels(
        ax,
        decimals=2,
    )

    save_figure(
        "02_gender_survival.png"
    )


# ============================================================
# 3. CLASS VS SURVIVAL
# ============================================================

def plot_class_survival(df):
    """
    Plot survival rate by passenger class.
    """

    validate_columns(
        df,
        {"Pclass", "Survived"},
    )

    plot_df = df.dropna(
        subset=["Pclass", "Survived"],
    ).copy()

    plot_df["Pclass"] = (
        plot_df["Pclass"].astype(int)
    )

    plt.figure(
        figsize=(10, 6)
    )

    ax = sns.barplot(
        data=plot_df,
        x="Pclass",
        y="Survived",
        hue="Pclass",
        order=[1, 2, 3],
        hue_order=[1, 2, 3],
        palette={
            1: COLORS["gold"],
            2: COLORS["blue"],
            3: COLORS["red"],
        },
        errorbar=None,
        legend=False,
    )

    ax.set_title(
        "Survival Rate by Passenger Class",
        fontsize=18,
        color=COLORS["navy"],
    )

    ax.set_xlabel(
        "Passenger Class"
    )

    ax.set_ylabel(
        "Survival Rate"
    )

    ax.set_ylim(0, 1)

    add_bar_labels(
        ax,
        decimals=2,
    )

    save_figure(
        "03_class_survival.png"
    )


# ============================================================
# 4. GENDER + CLASS
# ============================================================

def plot_gender_class_survival(df):
    """
    Plot survival rate by passenger class and gender.
    """

    validate_columns(
        df,
        {"Pclass", "Sex", "Survived"},
    )

    plot_df = df.dropna(
        subset=[
            "Pclass",
            "Sex",
            "Survived",
        ],
    ).copy()

    plot_df["Pclass"] = (
        plot_df["Pclass"].astype(int)
    )

    grouped = (
        plot_df
        .groupby(
            ["Pclass", "Sex"],
            as_index=False,
        )["Survived"]
        .mean()
    )

    plt.figure(
        figsize=(11, 7)
    )

    ax = sns.barplot(
        data=grouped,
        x="Pclass",
        y="Survived",
        hue="Sex",
        order=[1, 2, 3],
        hue_order=[
            "male",
            "female",
        ],
        palette={
            "male": COLORS["blue"],
            "female": COLORS["purple"],
        },
        errorbar=None,
    )

    ax.set_title(
        "Survival Rate by Passenger Class and Gender",
        fontsize=18,
        color=COLORS["navy"],
    )

    ax.set_xlabel(
        "Passenger Class"
    )

    ax.set_ylabel(
        "Survival Rate"
    )

    ax.set_ylim(0, 1)

    add_bar_labels(
        ax,
        decimals=2,
    )

    ax.legend(
        title="Gender",
        frameon=True,
    )

    save_figure(
        "04_gender_class_survival.png"
    )


# ============================================================
# 5. AGE DISTRIBUTION
# ============================================================

def plot_age_distribution(df):
    """
    Plot age distribution by survival status.
    """

    validate_columns(
        df,
        {"Age", "Survived"},
    )

    plot_df = df.dropna(
        subset=[
            "Age",
            "Survived",
        ],
    ).copy()

    plt.figure(
        figsize=(11, 6)
    )

    ax = sns.histplot(
        data=plot_df,
        x="Age",
        hue="Survived",
        bins=30,
        kde=True,
        hue_order=[0, 1],
        palette={
            0: COLORS["red"],
            1: COLORS["green"],
        },
        alpha=0.45,
    )

    ax.set_title(
        "Age Distribution by Survival Status",
        fontsize=18,
        color=COLORS["navy"],
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Passengers")

    save_figure(
        "05_age_distribution.png"
    )


# ============================================================
# 6. AGE BOX PLOT
# ============================================================

def plot_age_boxplot(df):
    """
    Plot age distribution by survival status.
    """

    validate_columns(
        df,
        {"Age", "Survived"},
    )

    plot_df = df.dropna(
        subset=[
            "Age",
            "Survived",
        ],
    ).copy()

    plt.figure(
        figsize=(10, 6)
    )

    ax = sns.boxplot(
        data=plot_df,
        x="Survived",
        y="Age",
        hue="Survived",
        order=[0, 1],
        hue_order=[0, 1],
        palette={
            0: COLORS["red"],
            1: COLORS["green"],
        },
        legend=False,
    )

    ax.set_title(
        "Age Distribution by Survival",
        fontsize=18,
        color=COLORS["navy"],
    )

    ax.set_xlabel(
        "Survival Status"
    )

    ax.set_ylabel("Age")

    ax.set_xticks([0, 1])

    ax.set_xticklabels(
        [
            "Did Not Survive",
            "Survived",
        ]
    )

    save_figure(
        "06_age_boxplot.png"
    )


# ============================================================
# 7. FARE
# ============================================================

def plot_fare_survival(df):
    """
    Plot fare distribution by passenger class and survival.
    """

    validate_columns(
        df,
        {
            "Pclass",
            "Fare",
            "Survived",
        },
    )

    plot_df = df.dropna(
        subset=[
            "Pclass",
            "Fare",
            "Survived",
        ],
    ).copy()

    plot_df["Pclass"] = (
        plot_df["Pclass"].astype(int)
    )

    fare_limit = (
        plot_df["Fare"].quantile(0.98)
    )

    plt.figure(
        figsize=(11, 6)
    )

    ax = sns.boxplot(
        data=plot_df,
        x="Pclass",
        y="Fare",
        hue="Survived",
        order=[1, 2, 3],
        hue_order=[0, 1],
        palette={
            0: COLORS["red"],
            1: COLORS["green"],
        },
    )

    ax.set_title(
        "Fare Distribution by Class and Survival",
        fontsize=18,
        color=COLORS["navy"],
    )

    ax.set_xlabel(
        "Passenger Class"
    )

    ax.set_ylabel("Fare")

    ax.set_ylim(
        0,
        fare_limit,
    )

    ax.legend(
        title="Survival",
        labels=[
            "Did Not Survive",
            "Survived",
        ],
    )

    save_figure(
        "07_fare_survival.png"
    )


# ============================================================
# 8. FAMILY SIZE
# ============================================================

def plot_family_size(df):
    """
    Plot survival rate by family size.

    FamilySize = SibSp + Parch + 1
    """

    validate_columns(
        df,
        {
            "SibSp",
            "Parch",
            "Survived",
        },
    )

    plot_df = df.dropna(
        subset=[
            "SibSp",
            "Parch",
            "Survived",
        ],
    ).copy()

    plot_df["FamilySize"] = (
        plot_df["SibSp"]
        + plot_df["Parch"]
        + 1
    )

    grouped = (
        plot_df
        .groupby(
            "FamilySize",
            as_index=False,
        )["Survived"]
        .mean()
    )

    plt.figure(
        figsize=(12, 6)
    )

    ax = sns.barplot(
        data=grouped,
        x="FamilySize",
        y="Survived",
        color=COLORS["blue"],
        errorbar=None,
    )

    ax.set_title(
        "Survival Rate by Family Size",
        fontsize=18,
        color=COLORS["navy"],
    )

    ax.set_xlabel(
        "Family Size"
    )

    ax.set_ylabel(
        "Survival Rate"
    )

    ax.set_ylim(0, 1)

    save_figure(
        "08_family_size_survival.png"
    )


# ============================================================
# 9. EMBARKED
# ============================================================

def plot_embarked_survival(df):
    """
    Plot survival rate by port of embarkation.

    S = Southampton
    C = Cherbourg
    Q = Queenstown
    """

    validate_columns(
        df,
        {
            "Embarked",
            "Survived",
        },
    )

    plot_df = df.dropna(
        subset=[
            "Embarked",
            "Survived",
        ],
    ).copy()

    embarked_order = [
        "S",
        "C",
        "Q",
    ]

    plt.figure(
        figsize=(10, 6)
    )

    ax = sns.barplot(
        data=plot_df,
        x="Embarked",
        y="Survived",
        hue="Embarked",
        order=embarked_order,
        hue_order=embarked_order,
        palette={
            "S": COLORS["blue"],
            "C": COLORS["gold"],
            "Q": COLORS["purple"],
        },
        errorbar=None,
        legend=False,
    )

    ax.set_title(
        "Survival Rate by Port of Embarkation",
        fontsize=18,
        color=COLORS["navy"],
    )

    ax.set_xlabel(
        "Embarkation Port"
    )

    ax.set_ylabel(
        "Survival Rate"
    )

    ax.set_ylim(0, 1)

    ax.set_xticks([0, 1, 2])

    ax.set_xticklabels(
        [
            "Southampton (S)",
            "Cherbourg (C)",
            "Queenstown (Q)",
        ]
    )

    add_bar_labels(
        ax,
        decimals=2,
    )

    save_figure(
        "09_embarked_survival.png"
    )


# ============================================================
# 10. CORRELATION
# ============================================================

def plot_correlation(df):
    """
    Plot correlation matrix for useful numeric features.

    PassengerId is excluded because it is an identifier.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    numeric = numeric.drop(
        columns=[
            "PassengerId",
        ],
        errors="ignore",
    )

    if numeric.empty:
        raise ValueError(
            "No numeric columns available "
            "for correlation analysis."
        )

    correlation = numeric.corr()

    plt.figure(
        figsize=(12, 9)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={
            "label": "Correlation"
        },
    )

    plt.title(
        "Numerical Feature Correlation Matrix",
        fontsize=18,
        color=COLORS["navy"],
    )

    save_figure(
        "10_correlation_heatmap.png"
    )


# ============================================================
# MASTER FUNCTION
# ============================================================

def generate_all_visualizations(df):
    """
    Generate all Titanic exploratory visualizations.

    Args:
        df (pd.DataFrame): Titanic training dataset.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    print("\n" + "=" * 70)
    print("TITANIC DATA VISUALIZATION")
    print("=" * 70)

    print("\nGenerating visualizations...")
    print("-" * 70)

    plot_survival_distribution(df)
    plot_gender_survival(df)
    plot_class_survival(df)
    plot_gender_class_survival(df)
    plot_age_distribution(df)
    plot_age_boxplot(df)
    plot_fare_survival(df)
    plot_family_size(df)
    plot_embarked_survival(df)
    plot_correlation(df)

    print("-" * 70)
    print(
        "All visualizations generated successfully."
    )

    print(
        f"Figures directory: {FIGURE_DIR}"
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

def main():
    """
    Load training data and generate all visualizations.
    """

    from .data_loader import load_train_data

    train = load_train_data()

    generate_all_visualizations(
        train
    )


if __name__ == "__main__":
    main()
