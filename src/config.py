import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.environ.get("DATA_PATH", str(ROOT / "data" / "AmesHousing.csv")))
MODEL_DIR = Path(os.environ.get("AIP_MODEL_DIR", str(ROOT / "models")).replace("gs://", "/gcs/"))
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# The approved model. A new model is copied here only if it passes the quality gate.
PRODUCTION_DIR = Path(
    os.environ.get("PRODUCTION_MODEL_DIR", str(ROOT / "models" / "production")).replace("gs://", "/gcs/")
)
PRODUCTION_MODEL_PATH = PRODUCTION_DIR / "model.joblib"
PRODUCTION_METRICS_PATH = PRODUCTION_DIR / "metrics.json"

# Quality gate
MAX_RMSE_LOG = 0.13     # absolute bar: never promote a model worse than this
GATE_TOLERANCE = 0.01   # new model may be up to 1% worse than production (test-score noise)

TARGET = "SalePrice"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CATEGORICAL_AS_STR = ["MS SubClass", "Mo Sold"]
QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

NONE_FILL_COLS = ["Pool QC", "Misc Feature", "Alley", "Fence", "Fireplace Qu", "Mas Vnr Type",
            "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
            "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2"]

ZERO_FILL_COLS = ["Total Bsmt SF", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
            "Bsmt Full Bath", "Bsmt Half Bath", "Garage Area", "Garage Cars", "Mas Vnr Area"]

DROP_COLS = ["Order", "PID", "Street", "Utilities", "Condition 2", "Roof Matl", "Heating",
            "Pool QC", "Misc Feature", "Alley", "Garage Area", "Garage Yr Blt", "TotRms AbvGrd"]

QUAL_COLS = ["Exter Qual", "Kitchen Qual", "Bsmt Qual", "Heating QC", "Fireplace Qu", "Garage Qual"]

BASELINE_FEATURES = ["Gr Liv Area", "Bedroom AbvGr", "total_bath", "2nd Flr SF"]

# Data validation: catches broken data, not unusual data (that is monitoring's job),
# so limits are physically plausible bounds, not this dataset's min/max.
MIN_TRAIN_ROWS = 500
VALID_QUAL_GRADES = {grade for grade in QUAL_MAP if grade != "None"}
VALUE_RANGES = {
    "Gr Liv Area": (100, 20_000),
    "Lot Area": (100, 1_000_000),
    "Overall Qual": (1, 10),
}
NON_ZERO_COLS = ["Gr Liv Area", "Lot Area"]
# Every raw column the code reads; dict.fromkeys removes duplicates and keeps order
REQUIRED_COLS = list(dict.fromkeys(
    CATEGORICAL_AS_STR + NONE_FILL_COLS + ZERO_FILL_COLS + QUAL_COLS + list(VALUE_RANGES)
    + ["Full Bath", "Half Bath", "Year Built", "Yr Sold"]
))
