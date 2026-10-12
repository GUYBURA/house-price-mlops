from src import config
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_bath"] = df["Full Bath"] + 0.5*df["Half Bath"] + df["Bsmt Full Bath"] + 0.5*df["Bsmt Half Bath"]
    df["age_at_sale"] = df["Yr Sold"] - df["Year Built"]
    df[config.QUAL_COLS] = df[config.QUAL_COLS].apply(lambda s: s.map(config.QUAL_MAP))
    df = df.drop(columns=config.DROP_COLS)
    return df

def split_target(df: pd.DataFrame):
    X = df.drop(columns=config.TARGET, errors="ignore")
    y = df[config.TARGET]
    return X, y