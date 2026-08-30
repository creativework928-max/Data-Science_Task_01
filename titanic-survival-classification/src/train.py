# import pandas as pd
# import numpy as np
# import joblib

# from sklearn.model_selection import (
#     train_test_split,
#     StratifiedKFold,
#     cross_validate
# )

# from sklearn.pipeline import Pipeline

# from sklearn.linear_model import LogisticRegression

# from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import (
#     RandomForestClassifier,
#     GradientBoostingClassifier,
#     HistGradientBoostingClassifier
# )

# from src.data_loader import load_train_data
# from src.feature_engineering import prepare_features
# from src.preprocessing import create_preprocessor

# from config import (
#     TARGET,
#     RANDOM_STATE,
#     MODEL_DIR
# )


# # ============================================================
# # LOAD DATA
# # ============================================================

# df = load_train_data()

# X = prepare_features(df)
# y = df[TARGET]


# print("\nFinal feature columns:")
# print(X.columns.tolist())

# print("\nFeature count:", X.shape[1])


# # ============================================================
# # TRAIN / VALIDATION SPLIT
# # ============================================================

# X_train, X_valid, y_train, y_valid = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=RANDOM_STATE,
#     stratify=y
# )


# # ============================================================
# # MODELS
# # ============================================================

# models = {

#     "Logistic Regression": LogisticRegression(
#         max_iter=3000,
#         class_weight="balanced",
#         random_state=RANDOM_STATE
#     ),

#     "Decision Tree": DecisionTreeClassifier(
#         max_depth=5,
#         min_samples_leaf=5,
#         class_weight="balanced",
#         random_state=RANDOM_STATE
#     ),

#     "Random Forest": RandomForestClassifier(
#         n_estimators=500,
#         max_depth=8,
#         min_samples_leaf=3,
#         max_features="sqrt",
#         class_weight="balanced",
#         random_state=RANDOM_STATE,
#         n_jobs=-1
#     ),

#     "Gradient Boosting": GradientBoostingClassifier(
#         n_estimators=300,
#         learning_rate=0.03,
#         max_depth=3,
#         min_samples_leaf=5,
#         random_state=RANDOM_STATE
#     ),

#     "HistGradient Boosting": HistGradientBoostingClassifier(
#         max_iter=300,
#         learning_rate=0.05,
#         max_leaf_nodes=15,
#         l2_regularization=1.0,
#         random_state=RANDOM_STATE
#     )
# }


# # ============================================================
# # CROSS VALIDATION
# # ============================================================

# cv = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=RANDOM_STATE
# )

# results = []


# for name, model in models.items():

#     print("\n" + "=" * 70)
#     print(name)
#     print("=" * 70)

#     preprocessor = create_preprocessor(X)

#     pipeline = Pipeline(
#         steps=[
#             (
#                 "preprocessing",
#                 preprocessor
#             ),
#             (
#                 "model",
#                 model
#             )
#         ]
#     )

#     scores = cross_validate(
#         pipeline,
#         X,
#         y,
#         cv=cv,
#         scoring=[
#             "accuracy",
#             "precision",
#             "recall",
#             "f1",
#             "roc_auc"
#         ],
#         n_jobs=-1
#     )

#     result = {
#         "Model": name,
#         "Accuracy": scores["test_accuracy"].mean(),
#         "Precision": scores["test_precision"].mean(),
#         "Recall": scores["test_recall"].mean(),
#         "F1": scores["test_f1"].mean(),
#         "ROC_AUC": scores["test_roc_auc"].mean()
#     }

#     results.append(result)

#     print(
#         f"Accuracy : {result['Accuracy']:.4f}"
#     )

#     print(
#         f"Precision: {result['Precision']:.4f}"
#     )

#     print(
#         f"Recall   : {result['Recall']:.4f}"
#     )

#     print(
#         f"F1       : {result['F1']:.4f}"
#     )

#     print(
#         f"ROC-AUC  : {result['ROC_AUC']:.4f}"
#     )


# # ============================================================
# # MODEL COMPARISON
# # ============================================================

# results_df = (
#     pd.DataFrame(results)
#     .sort_values(
#         "ROC_AUC",
#         ascending=False
#     )
# )

# print("\nFINAL MODEL COMPARISON")
# print(results_df.to_string(index=False))


# # ============================================================
# # SELECT BEST MODEL
# # ============================================================

# best_model_name = results_df.iloc[0]["Model"]

# print(
#     f"\nBest model according to ROC-AUC: "
#     f"{best_model_name}"
# )


# # ============================================================
# # FIT BEST MODEL ON COMPLETE DATA
# # ============================================================

# best_model = models[best_model_name]

# final_preprocessor = create_preprocessor(X)

# final_pipeline = Pipeline(
#     steps=[
#         (
#             "preprocessing",
#             final_preprocessor
#         ),
#         (
#             "model",
#             best_model
#         )
#     ]
# )

# final_pipeline.fit(X, y)


# # ============================================================
# # SAVE MODEL
# # ============================================================

# model_path = MODEL_DIR / "titanic_survival_model.joblib"

# joblib.dump(
#     final_pipeline,
#     model_path
# )

# print(
#     f"\nModel saved to: {model_path}"
# )


# # ============================================================
# # SAVE RESULTS
# # ============================================================

# results_df.to_csv(
#     "outputs/reports/model_comparison.csv",
#     index=False
# )




















































import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from .config import TARGET, RANDOM_STATE, MODEL_DIR, REPORT_DIR
from .data_loader import load_train_data
from .feature_engineering import prepare_features
from .preprocessing import create_preprocessor


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the Titanic training dataset and prepare features.

    Returns:
        tuple:
            X: Feature DataFrame.
            y: Target Series.
    """

    df = load_train_data()

    X = prepare_features(df)
    y = df[TARGET]

    return X, y


# ============================================================
# DEFINE MODELS
# ============================================================

def create_models():
    """
    Create the candidate classification models.

    Returns:
        dict: Dictionary containing model names and estimators.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),

        "HistGradient Boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    return models


# ============================================================
# CREATE PIPELINE
# ============================================================

def create_model_pipeline(model, X):
    """
    Create a preprocessing + model pipeline.

    Args:
        model: Scikit-learn estimator.
        X (pd.DataFrame): Feature DataFrame.

    Returns:
        Pipeline: Complete ML pipeline.
    """

    preprocessor = create_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_models(X, y, models):
    """
    Evaluate all candidate models using stratified cross-validation.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        models (dict): Candidate models.

    Returns:
        pd.DataFrame: Model comparison results.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    results = []

    for name, model in models.items():

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        pipeline = create_model_pipeline(
            model,
            X,
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        result = {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC_AUC": scores["test_roc_auc"].mean(),
        }

        results.append(result)

        print(f"Accuracy : {result['Accuracy']:.4f}")
        print(f"Precision: {result['Precision']:.4f}")
        print(f"Recall   : {result['Recall']:.4f}")
        print(f"F1       : {result['F1']:.4f}")
        print(f"ROC-AUC  : {result['ROC_AUC']:.4f}")

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            by="ROC_AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return results_df


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(X, y, models, best_model_name):
    """
    Train the selected model on the complete training dataset.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        models (dict): Candidate models.
        best_model_name (str): Name of selected model.

    Returns:
        Pipeline: Fitted final model pipeline.
    """

    best_model = models[best_model_name]

    final_pipeline = create_model_pipeline(
        best_model,
        X,
    )

    final_pipeline.fit(X, y)

    return final_pipeline


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """
    Save the trained model pipeline.

    Args:
        model: Fitted scikit-learn pipeline.

    Returns:
        Path: Saved model path.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "titanic_survival_model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    return model_path


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results_df):
    """
    Save model comparison results.

    Args:
        results_df (pd.DataFrame): Model comparison DataFrame.

    Returns:
        Path: Saved results path.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = REPORT_DIR / "model_comparison.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    return results_path


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run the complete model training workflow.
    """

    print("\n" + "=" * 70)
    print("TITANIC SURVIVAL MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load and prepare data
    # --------------------------------------------------------

    X, y = load_data()

    print("\nFinal feature columns:")
    print(X.columns.tolist())

    print(f"\nFeature count: {X.shape[1]}")
    print(f"Training rows: {X.shape[0]}")

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    models = create_models()

    print("\nModels to evaluate:")

    for model_name in models:
        print(f"  - {model_name}")

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    results_df = evaluate_models(
        X,
        y,
        models,
    )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    best_model_name = results_df.iloc[0]["Model"]

    print(
        f"\nBest model according to ROC-AUC: "
        f"{best_model_name}"
    )

    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    print("\nTraining final model on complete dataset...")

    final_pipeline = train_final_model(
        X,
        y,
        models,
        best_model_name,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = save_model(
        final_pipeline
    )

    print(
        f"\nModel saved to: {model_path}"
    )

    # --------------------------------------------------------
    # Save comparison results
    # --------------------------------------------------------

    results_path = save_results(
        results_df
    )

    print(
        f"Results saved to: {results_path}"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
