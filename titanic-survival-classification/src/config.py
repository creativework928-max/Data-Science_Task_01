from pathlib import Path

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = OUTPUT_DIR / "reports"

RANDOM_STATE = 42

TARGET = "Survived"

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"

# Professional visualization palette
COLORS = {
    "navy": "#0B1F3A",
    "blue": "#2563EB",
    "light_blue": "#60A5FA",
    "gold": "#F59E0B",
    "green": "#10B981",
    "red": "#EF4444",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "light_gray": "#E2E8F0",
    "white": "#FFFFFF",
    "background": "#F8FAFC",
}

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
