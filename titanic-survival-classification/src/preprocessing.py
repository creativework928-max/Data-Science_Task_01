# import pandas as pd

# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import (
#     OneHotEncoder,
#     StandardScaler
# )


# def create_preprocessor(X):

#     numeric_features = X.select_dtypes(
#         include=["int64", "float64", "int32", "float32"]
#     ).columns.tolist()

#     categorical_features = X.select_dtypes(
#         include=["object", "category", "bool"]
#     ).columns.tolist()

#     numeric_pipeline = Pipeline(
#         steps=[
#             (
#                 "imputer",
#                 SimpleImputer(
#                     strategy="median"
#                 )
#             ),
#             (
#                 "scaler",
#                 StandardScaler()
#             )
#         ]
#     )

#     categorical_pipeline = Pipeline(
#         steps=[
#             (
#                 "imputer",
#                 SimpleImputer(
#                     strategy="most_frequent"
#                 )
#             ),
#             (
#                 "onehot",
#                 OneHotEncoder(
#                     handle_unknown="ignore",
#                     sparse_output=False
#                 )
#             )
#         ]
#     )

#     preprocessor = ColumnTransformer(
#         transformers=[
#             (
#                 "numeric",
#                 numeric_pipeline,
#                 numeric_features
#             ),
#             (
#                 "categorical",
#                 categorical_pipeline,
#                 categorical_features
#             )
#         ],
#         remainder="drop"
#     )

#     return preprocessor























































import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(X):
    """
    Create a preprocessing pipeline for numerical and categorical features.

    Numerical features:
        - Missing values are filled using the median.
        - Features are standardized using StandardScaler.

    Categorical features:
        - Missing values are filled using the most frequent value.
        - Categories are converted to numerical features using one-hot encoding.

    Args:
        X (pd.DataFrame): Feature dataset used to identify column types.

    Returns:
        ColumnTransformer: Configured preprocessing pipeline.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    # Identify numerical columns
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Identify categorical columns
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical preprocessing pipeline
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Categorical preprocessing pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # Combine numerical and categorical preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor
