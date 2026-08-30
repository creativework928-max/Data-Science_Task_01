# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report,
#     RocCurveDisplay,
#     PrecisionRecallDisplay
# )

# from src.feature_engineering import prepare_features
# from src.data_loader import load_train_data

# from config import (
#     TARGET,
#     RANDOM_STATE,
#     FIGURE_DIR
# )


# # ============================================================
# # LOAD DATA
# # ============================================================

# df = load_train_data()

# X = prepare_features(df)
# y = df[TARGET]


# # ============================================================
# # SPLIT
# # ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=RANDOM_STATE,
#     stratify=y
# )


# # ============================================================
# # LOAD MODEL
# # ============================================================

# model = joblib.load(
#     "outputs/models/titanic_survival_model.joblib"
# )


# # ============================================================
# # PREDICTIONS
# # ============================================================

# y_pred = model.predict(X_test)

# y_prob = model.predict_proba(X_test)[:, 1]


# # ============================================================
# # METRICS
# # ============================================================

# metrics = {
#     "Accuracy": accuracy_score(
#         y_test,
#         y_pred
#     ),

#     "Precision": precision_score(
#         y_test,
#         y_pred
#     ),

#     "Recall": recall_score(
#         y_test,
#         y_pred
#     ),

#     "F1": f1_score(
#         y_test,
#         y_pred
#     ),

#     "ROC-AUC": roc_auc_score(
#         y_test,
#         y_prob
#     )
# }


# print("\nMODEL PERFORMANCE")
# print("=" * 50)

# for metric, value in metrics.items():

#     print(
#         f"{metric:12}: {value:.4f}"
#     )


# print("\nCLASSIFICATION REPORT")
# print(
#     classification_report(
#         y_test,
#         y_pred,
#         target_names=[
#             "Did Not Survive",
#             "Survived"
#         ]
#     )
# )


# # ============================================================
# # CONFUSION MATRIX
# # ============================================================

# cm = confusion_matrix(
#     y_test,
#     y_pred
# )

# plt.figure(figsize=(8, 6))

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     cbar=False,
#     xticklabels=[
#         "Predicted No",
#         "Predicted Yes"
#     ],
#     yticklabels=[
#         "Actual No",
#         "Actual Yes"
#     ]
# )

# plt.title(
#     "Titanic Survival Confusion Matrix",
#     fontsize=17,
#     fontweight="bold"
# )

# plt.xlabel("Prediction")
# plt.ylabel("Actual")

# plt.tight_layout()

# plt.savefig(
#     FIGURE_DIR / "11_confusion_matrix.png",
#     dpi=300
# )

# plt.close()


# # ============================================================
# # ROC CURVE
# # ============================================================

# fig, ax = plt.subplots(
#     figsize=(9, 7)
# )

# RocCurveDisplay.from_predictions(
#     y_test,
#     y_prob,
#     ax=ax,
#     color="#2563EB"
# )

# ax.set_title(
#     "ROC Curve - Titanic Survival Model",
#     fontsize=17,
#     fontweight="bold"
# )

# plt.tight_layout()

# plt.savefig(
#     FIGURE_DIR / "12_roc_curve.png",
#     dpi=300
# )

# plt.close()


# # ============================================================
# # PRECISION-RECALL CURVE
# # ============================================================

# fig, ax = plt.subplots(
#     figsize=(9, 7)
# )

# PrecisionRecallDisplay.from_predictions(
#     y_test,
#     y_prob,
#     ax=ax,
#     color="#7C3AED"
# )

# ax.set_title(
#     "Precision-Recall Curve",
#     fontsize=17,
#     fontweight="bold"
# )

# plt.tight_layout()

# plt.savefig(
#     FIGURE_DIR / "13_precision_recall_curve.png",
#     dpi=300
# )

# plt.close()





















































import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .config import (
    TARGET,
    RANDOM_STATE,
    FIGURE_DIR,
)

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
# CREATE MODEL
# ============================================================

def create_model_pipeline(X):
    """
    Create the Gradient Boosting preprocessing and model pipeline.

    Args:
        X (pd.DataFrame): Feature DataFrame.

    Returns:
        Pipeline: Complete ML pipeline.
    """

    preprocessor = create_preprocessor(X)

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )

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
# GENERATE OUT-OF-FOLD PREDICTIONS
# ============================================================

def generate_oof_predictions(X, y):
    """
    Generate out-of-fold predictions using stratified
    cross-validation.

    Each prediction is generated by a model that did not
    train on that particular observation.

    Returns:
        tuple:
            y_pred: Out-of-fold class predictions.
            y_prob: Out-of-fold survival probabilities.
    """

    pipeline = create_model_pipeline(X)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\nGenerating out-of-fold predictions...")

    y_pred = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=-1,
    )

    y_prob = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return y_pred, y_prob


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(y, y_pred, y_prob):
    """
    Calculate classification performance metrics.

    Returns:
        dict: Calculated metrics.
    """

    metrics = {
        "Accuracy": accuracy_score(
            y,
            y_pred,
        ),

        "Precision": precision_score(
            y,
            y_pred,
            zero_division=0,
        ),

        "Recall": recall_score(
            y,
            y_pred,
            zero_division=0,
        ),

        "F1": f1_score(
            y,
            y_pred,
            zero_division=0,
        ),

        "ROC-AUC": roc_auc_score(
            y,
            y_prob,
        ),
    }

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(metrics):
    """
    Print model performance metrics.
    """

    print("\nMODEL PERFORMANCE")
    print("=" * 50)

    for metric, value in metrics.items():
        print(
            f"{metric:12}: {value:.4f}"
        )


# ============================================================
# PRINT CLASSIFICATION REPORT
# ============================================================

def print_classification_report(y, y_pred):
    """
    Print the classification report.
    """

    print("\nCLASSIFICATION REPORT")
    print("=" * 50)

    print(
        classification_report(
            y,
            y_pred,
            target_names=[
                "Did Not Survive",
                "Survived",
            ],
            zero_division=0,
        )
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(y, y_pred):
    """
    Create and save the confusion matrix.
    """

    cm = confusion_matrix(
        y,
        y_pred,
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "Predicted No",
            "Predicted Yes",
        ],
        yticklabels=[
            "Actual No",
            "Actual Yes",
        ],
    )

    plt.title(
        "Titanic Survival Confusion Matrix",
        fontsize=17,
        fontweight="bold",
    )

    plt.xlabel("Prediction")
    plt.ylabel("Actual")

    plt.tight_layout()

    output_file = (
        FIGURE_DIR /
        "11_confusion_matrix.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Confusion matrix saved to: {output_file}"
    )


# ============================================================
# ROC CURVE
# ============================================================

# def save_roc_curve(y, y_prob):
#     """
#     Create and save the ROC curve.
#     """

#     fig, ax = plt.subplots(
#         figsize=(9, 7)
#     )

#     RocCurveDisplay.from_predictions(
#         y,
#         y_prob,
#         ax=ax,
#         color="#2563EB",
#     )

#     ax.set_title(
#         "ROC Curve - Titanic Survival Model",
#         fontsize=17,
#         fontweight="bold",
#     )

#     plt.tight_layout()

#     output_file = (
#         FIGURE_DIR /
#         "12_roc_curve.png"
#     )

#     plt.savefig(
#         output_file,
#         dpi=300,
#         bbox_inches="tight",
#     )

#     plt.close()

#     print(
#         f"ROC curve saved to: {output_file}"
#     )





def save_roc_curve(y, y_prob):
    """
    Create and save the ROC curve.
    """

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    RocCurveDisplay.from_predictions(
        y,
        y_prob,
        ax=ax,
    )

    # Set color after the curve has been created.
    if ax.lines:
        ax.lines[0].set_color("#2563EB")
        ax.lines[0].set_linewidth(2.5)

    ax.set_title(
        "ROC Curve - Titanic Survival Model",
        fontsize=17,
        fontweight="bold",
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_file = (
        FIGURE_DIR /
        "12_roc_curve.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"ROC curve saved to: {output_file}"
    )


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

# def save_precision_recall_curve(y, y_prob):
#     """
#     Create and save the Precision-Recall curve.
#     """

#     fig, ax = plt.subplots(
#         figsize=(9, 7)
#     )

#     PrecisionRecallDisplay.from_predictions(
#         y,
#         y_prob,
#         ax=ax,
#         color="#7C3AED",
#     )

#     ax.set_title(
#         "Precision-Recall Curve - Titanic Survival Model",
#         fontsize=17,
#         fontweight="bold",
#     )

#     plt.tight_layout()

#     output_file = (
#         FIGURE_DIR /
#         "13_precision_recall_curve.png"
#     )

#     plt.savefig(
#         output_file,
#         dpi=300,
#         bbox_inches="tight",
#     )

#     plt.close()

#     print(
#         f"Precision-Recall curve saved to: {output_file}"
#     )




def save_precision_recall_curve(y, y_prob):
    """
    Create and save the Precision-Recall curve.
    """

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    PrecisionRecallDisplay.from_predictions(
        y,
        y_prob,
        ax=ax,
    )

    # Set color after the curve has been created.
    if ax.lines:
        ax.lines[0].set_color("#7C3AED")
        ax.lines[0].set_linewidth(2.5)

    ax.set_title(
        "Precision-Recall Curve - Titanic Survival Model",
        fontsize=17,
        fontweight="bold",
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_file = (
        FIGURE_DIR /
        "13_precision_recall_curve.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Precision-Recall curve saved to: {output_file}"
    )



# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run the complete model evaluation workflow.
    """

    print("\n" + "=" * 70)
    print("TITANIC SURVIVAL MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y = load_data()

    print(
        f"\nDataset rows: {X.shape[0]}"
    )

    print(
        f"Feature count: {X.shape[1]}"
    )

    # --------------------------------------------------------
    # Generate out-of-fold predictions
    # --------------------------------------------------------

    y_pred, y_prob = generate_oof_predictions(
        X,
        y,
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y,
        y_pred,
        y_prob,
    )

    # --------------------------------------------------------
    # Display metrics
    # --------------------------------------------------------

    print_metrics(metrics)

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print_classification_report(
        y,
        y_pred,
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save visualizations
    # --------------------------------------------------------

    save_confusion_matrix(
        y,
        y_pred,
    )

    save_roc_curve(
        y,
        y_prob,
    )

    save_precision_recall_curve(
        y,
        y_prob,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
