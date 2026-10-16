import joblib

from src.config import MODEL_PATH, TARGET
from src.data import clean_data, load_data
from src.features import add_features
from src.validate import validate


def predict(df):
    model = joblib.load(MODEL_PATH)
    validate(df, require_target=False)
    df = clean_data(df)
    df = add_features(df)
    X  = df.drop(columns=TARGET, errors="ignore")
    return model.predict(X)

if __name__  ==  "__main__":
    sample = load_data().head(5)
    preds = predict(sample)
    for real, pred in zip(sample["SalePrice"], preds):
        print(f"real {real:>9,.0f}    predicted {pred:>9,.0f}")
