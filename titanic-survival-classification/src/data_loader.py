# import pandas as pd

# from config import TRAIN_FILE, TEST_FILE


# def load_train_data():
#     """
#     Load Titanic training data.
#     """

#     df = pd.read_csv(TRAIN_FILE)

#     print("=" * 70)
#     print("TRAINING DATA")
#     print("=" * 70)

#     print(f"Rows    : {df.shape[0]}")
#     print(f"Columns : {df.shape[1]}")

#     return df


# def load_test_data():
#     """
#     Load Titanic test data.
#     """

#     df = pd.read_csv(TEST_FILE)

#     print("=" * 70)
#     print("TEST DATA")
#     print("=" * 70)

#     print(f"Rows    : {df.shape[0]}")
#     print(f"Columns : {df.shape[1]}")

#     return df


# def validate_data(train, test):
#     """
#     Basic dataset validation.
#     """

#     required_train_columns = {
#         "PassengerId",
#         "Survived",
#         "Pclass",
#         "Name",
#         "Sex",
#         "Age",
#         "SibSp",
#         "Parch",
#         "Ticket",
#         "Fare",
#         "Cabin",
#         "Embarked",
#     }

#     required_test_columns = required_train_columns - {"Survived"}

#     missing_train = required_train_columns - set(train.columns)
#     missing_test = required_test_columns - set(test.columns)

#     if missing_train:
#         raise ValueError(
#             f"Missing training columns: {missing_train}"
#         )

#     if missing_test:
#         raise ValueError(
#             f"Missing test columns: {missing_test}"
#         )

#     print("\nData validation successful.")


# if __name__ == "__main__":

#     train = load_train_data()
#     test = load_test_data()

#     validate_data(train, test)


























































# import pandas as pd

# from config import TRAIN_FILE, TEST_FILE


# def load_train_data():
#     """
#     Load the Titanic training dataset.

#     Returns:
#         pd.DataFrame: Training dataset.
#     """
#     try:
#         df = pd.read_csv(TRAIN_FILE)
#     except FileNotFoundError:
#         raise FileNotFoundError(
#             f"Training file not found: {TRAIN_FILE}"
#         )
#     except Exception as exc:
#         raise RuntimeError(
#             f"Failed to load training data from {TRAIN_FILE}: {exc}"
#         )

#     print("=" * 70)
#     print("TRAINING DATA")
#     print("=" * 70)
#     print(f"Rows    : {df.shape[0]}")
#     print(f"Columns : {df.shape[1]}")

#     return df


# def load_test_data():
#     """
#     Load the Titanic test dataset.

#     Returns:
#         pd.DataFrame: Test dataset.
#     """
#     try:
#         df = pd.read_csv(TEST_FILE)
#     except FileNotFoundError:
#         raise FileNotFoundError(
#             f"Test file not found: {TEST_FILE}"
#         )
#     except Exception as exc:
#         raise RuntimeError(
#             f"Failed to load test data from {TEST_FILE}: {exc}"
#         )

#     print("=" * 70)
#     print("TEST DATA")
#     print("=" * 70)
#     print(f"Rows    : {df.shape[0]}")
#     print(f"Columns : {df.shape[1]}")

#     return df


# def validate_data(train, test):
#     """
#     Perform basic validation on the training and test datasets.

#     Args:
#         train (pd.DataFrame): Training dataset.
#         test (pd.DataFrame): Test dataset.

#     Raises:
#         TypeError: If train or test is not a pandas DataFrame.
#         ValueError: If required columns are missing or duplicate columns exist.
#     """

#     if not isinstance(train, pd.DataFrame):
#         raise TypeError("train must be a pandas DataFrame.")

#     if not isinstance(test, pd.DataFrame):
#         raise TypeError("test must be a pandas DataFrame.")

#     required_train_columns = {
#         "PassengerId",
#         "Survived",
#         "Pclass",
#         "Name",
#         "Sex",
#         "Age",
#         "SibSp",
#         "Parch",
#         "Ticket",
#         "Fare",
#         "Cabin",
#         "Embarked",
#     }

#     required_test_columns = required_train_columns - {"Survived"}

#     # Check for duplicate column names
#     duplicate_train_columns = train.columns[
#         train.columns.duplicated()
#     ].tolist()

#     duplicate_test_columns = test.columns[
#         test.columns.duplicated()
#     ].tolist()

#     if duplicate_train_columns:
#         raise ValueError(
#             f"Duplicate training columns found: "
#             f"{duplicate_train_columns}"
#         )

#     if duplicate_test_columns:
#         raise ValueError(
#             f"Duplicate test columns found: "
#             f"{duplicate_test_columns}"
#         )

#     # Check required training columns
#     missing_train = required_train_columns - set(train.columns)

#     if missing_train:
#         raise ValueError(
#             f"Missing training columns: {sorted(missing_train)}"
#         )

#     # Check required test columns
#     missing_test = required_test_columns - set(test.columns)

#     if missing_test:
#         raise ValueError(
#             f"Missing test columns: {sorted(missing_test)}"
#         )

#     print("\nData validation successful.")


# def main():
#     """
#     Load and validate the Titanic datasets.
#     """
#     train = load_train_data()
#     test = load_test_data()

#     validate_data(train, test)

#     return train, test


# if __name__ == "__main__":
#     main()





















































import pandas as pd

from .config import TRAIN_FILE, TEST_FILE


def load_train_data():
    """
    Load the Titanic training dataset.

    Returns:
        pd.DataFrame: Training dataset.
    """
    try:
        df = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load training data from {TRAIN_FILE}: {exc}"
        ) from exc

    print("=" * 70)
    print("TRAINING DATA")
    print("=" * 70)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


def load_test_data():
    """
    Load the Titanic test dataset.

    Returns:
        pd.DataFrame: Test dataset.
    """
    try:
        df = pd.read_csv(TEST_FILE)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Test file not found: {TEST_FILE}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load test data from {TEST_FILE}: {exc}"
        ) from exc

    print("=" * 70)
    print("TEST DATA")
    print("=" * 70)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


def validate_data(train, test):
    """
    Perform basic validation on the training and test datasets.

    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset.

    Raises:
        TypeError: If train or test is not a pandas DataFrame.
        ValueError: If required columns are missing or duplicate columns exist.
    """

    if not isinstance(train, pd.DataFrame):
        raise TypeError("train must be a pandas DataFrame.")

    if not isinstance(test, pd.DataFrame):
        raise TypeError("test must be a pandas DataFrame.")

    # Required columns for the training dataset
    required_train_columns = {
        "PassengerId",
        "Survived",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    }

    # Required columns for the test dataset
    # Test data does not contain "Survived"
    required_test_columns = required_train_columns - {"Survived"}

    # Check for duplicate training column names
    duplicate_train_columns = train.columns[
        train.columns.duplicated()
    ].tolist()

    if duplicate_train_columns:
        raise ValueError(
            f"Duplicate training columns found: "
            f"{duplicate_train_columns}"
        )

    # Check for duplicate test column names
    duplicate_test_columns = test.columns[
        test.columns.duplicated()
    ].tolist()

    if duplicate_test_columns:
        raise ValueError(
            f"Duplicate test columns found: "
            f"{duplicate_test_columns}"
        )

    # Check required training columns
    missing_train = required_train_columns - set(train.columns)

    if missing_train:
        raise ValueError(
            f"Missing training columns: {sorted(missing_train)}"
        )

    # Check required test columns
    missing_test = required_test_columns - set(test.columns)

    if missing_test:
        raise ValueError(
            f"Missing test columns: {sorted(missing_test)}"
        )

    print("\nData validation successful.")


def main():
    """
    Load and validate the Titanic datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Validated training and test datasets.
    """
    train = load_train_data()
    test = load_test_data()

    validate_data(train, test)

    return train, test


if __name__ == "__main__":
    main()
