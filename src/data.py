import pandas as pd

from src import config


def load_data(path= config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.CATEGORICAL_AS_STR] = df[config.CATEGORICAL_AS_STR].astype(str)
    df[config.NONE_FILL_COLS] = df[config.NONE_FILL_COLS].fillna("None")
    df[config.ZERO_FILL_COLS] = df[config.ZERO_FILL_COLS].fillna(0)
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~((df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 300000))].reset_index(drop=True)
    return df
