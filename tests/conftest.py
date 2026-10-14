import pandas as pd
import pytest

from src.data import load_data, clean_data, remove_outliers
from src.features import add_features, split_target


@pytest.fixture(scope="session")
def raw_data() -> pd.DataFrame:
    """Full raw CSV, loaded once per test run."""
    return load_data()


@pytest.fixture
def raw(raw_data) -> pd.DataFrame:
    """Fresh copy for each test, so no test can affect another."""
    return raw_data.copy()


@pytest.fixture(scope="session")
def train_xy(raw_data):
    """X, y prepared exactly like train.py does it."""
    df = add_features(remove_outliers(clean_data(raw_data)))
    return split_target(df)


@pytest.fixture(scope="session")
def small_xy(train_xy):
    """First 300 rows, enough for fast fit/predict smoke tests."""
    X, y = train_xy
    return X.head(300), y.head(300)
