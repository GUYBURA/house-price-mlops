import joblib
import pandas as pd

from src.config import PRODUCTION_MODEL_PATH, TARGET
from src.data import clean_data, load_data
from src.features import add_features
from src.validate import validate


def predict_with_model(model, df: pd.DataFrame):
    """The one prediction path, shared by this script and the serving app."""
    validate(df, require_target=False)
    df = clean_data(df)
    df = add_features(df)
    X = df.drop(columns=TARGET, errors="ignore")
    return model.predict(X)


def predict(df: pd.DataFrame):
    model = joblib.load(PRODUCTION_MODEL_PATH)
    return predict_with_model(model, df)


if __name__ == "__main__":
    sample = load_data().head(5)
    preds = predict(sample)
    for real, pred in zip(sample["SalePrice"], preds):
        print(f"real {real:>9,.0f}    predicted {pred:>9,.0f}")
