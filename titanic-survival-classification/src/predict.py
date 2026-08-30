# import pandas as pd
# import joblib

# from src.feature_engineering import prepare_features
# from config import TEST_FILE, PREDICTION_DIR


# # ============================================================
# # LOAD TEST DATA
# # ============================================================

# test = pd.read_csv(TEST_FILE)


# # ============================================================
# # PREPARE FEATURES
# # ============================================================

# X_test = prepare_features(test)


# # ============================================================
# # LOAD MODEL
# # ============================================================

# model = joblib.load(
#     "outputs/models/titanic_survival_model.joblib"
# )


# # ============================================================
# # PREDICT
# # ============================================================

# predictions = model.predict(X_test)


# # ============================================================
# # CREATE SUBMISSION
# # ============================================================

# submission = pd.DataFrame({
#     "PassengerId": test["PassengerId"],
#     "Survived": predictions.astype(int)
# })


# # ============================================================
# # SAVE
# # ============================================================

# output_file = (
#     PREDICTION_DIR /
#     "titanic_submission.csv"
# )

# submission.to_csv(
#     output_file,
#     index=False
# )


# print("=" * 60)
# print("KAGGLE SUBMISSION CREATED")
# print("=" * 60)

# print(
#     submission.head(10)
# )

# print(
#     f"\nRows: {len(submission)}"
# )

# print(
#     f"Saved: {output_file}"
# )






















































import joblib
import pandas as pd

from .config import TEST_FILE, PREDICTION_DIR, MODEL_DIR
from .feature_engineering import prepare_features


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    """
    Load the Titanic test dataset.

    Returns:
        pd.DataFrame: Test dataset.
    """

    try:
        test = pd.read_csv(TEST_FILE)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Test file not found: {TEST_FILE}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load test data from {TEST_FILE}: {exc}"
        ) from exc

    return test


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():
    """
    Load the trained Titanic survival model.

    Returns:
        object: Trained scikit-learn pipeline.
    """

    model_path = MODEL_DIR / "titanic_survival_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run 'python -m src.train' first."
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {model_path}: {exc}"
        ) from exc

    return model


# ============================================================
# MAKE PREDICTIONS
# ============================================================

def make_predictions(model, X_test):
    """
    Generate survival predictions and probabilities.

    Args:
        model: Trained model pipeline.
        X_test (pd.DataFrame): Test features.

    Returns:
        tuple:
            predictions: Predicted survival classes.
            probabilities: Probability of survival.
    """

    predictions = model.predict(X_test).astype(int)

    # Gradient Boosting and the other classifiers used in
    # this project support predict_proba().
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = None

    return predictions, probabilities


# ============================================================
# CREATE SUBMISSION
# ============================================================

def create_submission(
    test,
    predictions,
    probabilities=None,
):
    """
    Create the Kaggle-style Titanic submission DataFrame.

    Args:
        test (pd.DataFrame): Original test dataset.
        predictions: Predicted survival classes.
        probabilities: Optional survival probabilities.

    Returns:
        pd.DataFrame: Submission DataFrame.
    """

    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"],
            "Survived": predictions,
        }
    )

    # Keep probability in a separate analysis file rather than
    # adding it to the Kaggle submission.
    if probabilities is not None:
        submission["SurvivalProbability"] = probabilities

    return submission


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(submission):
    """
    Save predictions to the predictions directory.

    Args:
        submission (pd.DataFrame): Prediction DataFrame.

    Returns:
        Path: Saved prediction file.
    """

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        PREDICTION_DIR /
        "titanic_predictions.csv"
    )

    submission.to_csv(
        output_file,
        index=False,
    )

    return output_file


def save_kaggle_submission(submission):
    """
    Save the Kaggle-compatible submission file.

    Only PassengerId and Survived are included.

    Args:
        submission (pd.DataFrame): Prediction DataFrame.

    Returns:
        Path: Saved Kaggle submission file.
    """

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    kaggle_file = (
        PREDICTION_DIR /
        "titanic_submission.csv"
    )

    kaggle_submission = submission[
        [
            "PassengerId",
            "Survived",
        ]
    ]

    kaggle_submission.to_csv(
        kaggle_file,
        index=False,
    )

    return kaggle_file


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run the complete Titanic prediction workflow.
    """

    print("\n" + "=" * 70)
    print("TITANIC SURVIVAL PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test = load_test_data()

    print(f"\nTest rows: {len(test)}")

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_test = prepare_features(test)

    print(
        f"Test feature count: {X_test.shape[1]}"
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_model()

    print("\nTrained model loaded successfully.")

    # --------------------------------------------------------
    # Make predictions
    # --------------------------------------------------------

    predictions, probabilities = make_predictions(
        model,
        X_test,
    )

    # --------------------------------------------------------
    # Create submission
    # --------------------------------------------------------

    submission = create_submission(
        test,
        predictions,
        probabilities,
    )

    # --------------------------------------------------------
    # Save detailed predictions
    # --------------------------------------------------------

    prediction_file = save_predictions(
        submission
    )

    # --------------------------------------------------------
    # Save Kaggle submission
    # --------------------------------------------------------

    kaggle_file = save_kaggle_submission(
        submission
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)

    print("\nFirst 10 predictions:")
    print(
        submission.head(10).to_string(
            index=False
        )
    )

    print(
        f"\nSurvived predictions: "
        f"{predictions.sum()}"
    )

    print(
        f"Not survived predictions: "
        f"{len(predictions) - predictions.sum()}"
    )

    print(
        f"\nDetailed predictions saved to:"
        f"\n{prediction_file}"
    )

    print(
        f"\nKaggle submission saved to:"
        f"\n{kaggle_file}"
    )


if __name__ == "__main__":
    main()
