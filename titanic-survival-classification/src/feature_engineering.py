# import pandas as pd
# import numpy as np


# # ============================================================
# # FEATURE ENGINEERING
# # ============================================================

# def engineer_features(df):
#     """
#     Create meaningful Titanic survival features.
#     """

#     data = df.copy()

#     # --------------------------------------------------------
#     # 1. FAMILY SIZE
#     # --------------------------------------------------------

#     data["FamilySize"] = (
#         data["SibSp"] +
#         data["Parch"] +
#         1
#     )

#     # --------------------------------------------------------
#     # 2. IS ALONE
#     # --------------------------------------------------------

#     data["IsAlone"] = (
#         data["FamilySize"] == 1
#     ).astype(int)

#     # --------------------------------------------------------
#     # 3. MOTHER INDICATOR
#     # --------------------------------------------------------

#     data["IsMother"] = (
#         (data["Sex"] == "female") &
#         (data["Parch"] > 0) &
#         (data["Age"] > 18)
#     ).astype(int)

#     # --------------------------------------------------------
#     # 4. AGE GROUP
#     # --------------------------------------------------------

#     data["AgeGroup"] = pd.cut(
#         data["Age"],
#         bins=[
#             -np.inf,
#             5,
#             12,
#             18,
#             30,
#             45,
#             60,
#             np.inf
#         ],
#         labels=[
#             "Infant",
#             "Child",
#             "Teen",
#             "YoungAdult",
#             "Adult",
#             "MiddleAge",
#             "Senior"
#         ]
#     )

#     # --------------------------------------------------------
#     # 5. FARE PER PERSON
#     # --------------------------------------------------------

#     ticket_counts = (
#         data["Ticket"]
#         .map(data["Ticket"].value_counts())
#     )

#     data["TicketGroupSize"] = ticket_counts

#     data["FarePerPerson"] = (
#         data["Fare"] /
#         data["TicketGroupSize"].replace(0, 1)
#     )

#     # --------------------------------------------------------
#     # 6. CABIN DECK
#     # --------------------------------------------------------

#     data["Deck"] = (
#         data["Cabin"]
#         .fillna("Unknown")
#         .astype(str)
#         .str[0]
#     )

#     # --------------------------------------------------------
#     # 7. NUMBER OF CABINS
#     # --------------------------------------------------------

#     data["CabinKnown"] = (
#         data["Cabin"].notna()
#     ).astype(int)

#     # --------------------------------------------------------
#     # 8. TITLE
#     # --------------------------------------------------------

#     data["Title"] = (
#         data["Name"]
#         .str.extract(r",\s*([^.]*)\.", expand=False)
#         .str.strip()
#     )

#     # Group rare titles
#     common_titles = [
#         "Mr",
#         "Miss",
#         "Mrs",
#         "Master"
#     ]

#     data["Title"] = data["Title"].where(
#         data["Title"].isin(common_titles),
#         "Rare"
#     )

#     # --------------------------------------------------------
#     # 9. TICKET PREFIX
#     # --------------------------------------------------------

#     data["TicketPrefix"] = (
#         data["Ticket"]
#         .astype(str)
#         .str.replace(r"\d", "", regex=True)
#         .str.replace(r"[\s./]+", "", regex=True)
#         .replace("", "NONE")
#     )

#     # --------------------------------------------------------
#     # 10. FAMILY / SOCIAL GROUP
#     # --------------------------------------------------------

#     data["SmallFamily"] = (
#         data["FamilySize"].between(2, 4)
#     ).astype(int)

#     data["LargeFamily"] = (
#         data["FamilySize"] >= 5
#     ).astype(int)

#     return data


# def prepare_features(df):
#     """
#     Feature engineering followed by dropping fields that
#     should not directly enter the ML model.
#     """

#     data = engineer_features(df)

#     columns_to_drop = [
#         "Survived",
#         "PassengerId",
#         "Name",
#         "Ticket",
#         "Cabin"
#     ]

#     columns_to_drop = [
#         col for col in columns_to_drop
#         if col in data.columns
#     ]

#     X = data.drop(
#         columns=columns_to_drop,
#         errors="ignore"
#     )

#     return X























































import pandas as pd


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Create meaningful features for Titanic survival prediction.

    Args:
        df (pd.DataFrame): Raw Titanic dataset.

    Returns:
        pd.DataFrame: Dataset with engineered features.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    data = df.copy()

    # --------------------------------------------------------
    # 1. FAMILY SIZE
    # --------------------------------------------------------

    data["FamilySize"] = (
        data["SibSp"] +
        data["Parch"] +
        1
    )

    # --------------------------------------------------------
    # 2. IS ALONE
    # --------------------------------------------------------

    data["IsAlone"] = (
        data["FamilySize"] == 1
    ).astype(int)

    # --------------------------------------------------------
    # 3. MOTHER INDICATOR
    # --------------------------------------------------------

    data["IsMother"] = (
        (data["Sex"] == "female") &
        (data["Parch"] > 0) &
        (data["Age"] > 18)
    ).astype(int)

    # --------------------------------------------------------
    # 4. AGE GROUP
    # --------------------------------------------------------

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[
            float("-inf"),
            5,
            12,
            18,
            30,
            45,
            60,
            float("inf"),
        ],
        labels=[
            "Infant",
            "Child",
            "Teen",
            "YoungAdult",
            "Adult",
            "MiddleAge",
            "Senior",
        ],
    )

    # --------------------------------------------------------
    # 5. TICKET GROUP SIZE
    # --------------------------------------------------------

    # Number of passengers sharing the same ticket.
    data["TicketGroupSize"] = (
        data["Ticket"]
        .map(data["Ticket"].value_counts())
    )

    # --------------------------------------------------------
    # 6. FARE PER PERSON
    # --------------------------------------------------------

    data["FarePerPerson"] = (
        data["Fare"] /
        data["TicketGroupSize"].replace(0, 1)
    )

    # --------------------------------------------------------
    # 7. CABIN DECK
    # --------------------------------------------------------

    data["Deck"] = (
        data["Cabin"]
        .fillna("Unknown")
        .astype(str)
        .str[0]
    )

    # --------------------------------------------------------
    # 8. CABIN KNOWN
    # --------------------------------------------------------

    data["CabinKnown"] = (
        data["Cabin"].notna()
    ).astype(int)

    # --------------------------------------------------------
    # 9. TITLE
    # --------------------------------------------------------

    data["Title"] = (
        data["Name"]
        .astype(str)
        .str.extract(
            r",\s*([^.]*)\.",
            expand=False,
        )
        .str.strip()
    )

    # Group uncommon titles into "Rare"
    common_titles = {
        "Mr",
        "Miss",
        "Mrs",
        "Master",
    }

    data["Title"] = data["Title"].where(
        data["Title"].isin(common_titles),
        "Rare",
    )

    # --------------------------------------------------------
    # 10. TICKET PREFIX
    # --------------------------------------------------------

    data["TicketPrefix"] = (
        data["Ticket"]
        .astype(str)
        .str.replace(r"\d", "", regex=True)
        .str.replace(r"[\s./]+", "", regex=True)
        .replace("", "NONE")
    )

    # --------------------------------------------------------
    # 11. FAMILY SIZE GROUPS
    # --------------------------------------------------------

    data["SmallFamily"] = (
        data["FamilySize"].between(2, 4)
    ).astype(int)

    data["LargeFamily"] = (
        data["FamilySize"] >= 5
    ).astype(int)

    return data


# ============================================================
# PREPARE MODEL FEATURES
# ============================================================

def prepare_features(df):
    """
    Apply feature engineering and remove columns that should
    not directly be used by the machine-learning model.

    Args:
        df (pd.DataFrame): Raw Titanic dataset.

    Returns:
        pd.DataFrame: Model-ready feature DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    data = engineer_features(df)

    columns_to_drop = [
        "Survived",
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin",
    ]

    # Only drop columns that actually exist.
    columns_to_drop = [
        column
        for column in columns_to_drop
        if column in data.columns
    ]

    X = data.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    return X
