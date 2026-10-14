import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.environ.get("DATA_PATH", str(ROOT / "data" / "AmesHousing.csv")))
MODEL_DIR = Path(os.environ.get("AIP_MODEL_DIR", str(ROOT / "models")).replace("gs://", "/gcs/"))
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

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