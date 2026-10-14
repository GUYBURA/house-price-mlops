import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor():
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one-hot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformer = ColumnTransformer([
        ("numerical", numerical_pipeline, make_column_selector(dtype_include="number")),
        ("categorical", categorical_pipeline, make_column_selector(dtype_exclude="number"))
    ], sparse_threshold=0)
    return transformer


def build_pipeline(model=None):
    if model is None:
        model = Ridge(alpha=10)
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", model)
    ])
    model_with_log_target = TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1
    )
    return model_with_log_target
