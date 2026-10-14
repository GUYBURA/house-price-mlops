import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.pipeline import build_pipeline, build_preprocessor


def test_preprocessor_output_has_no_missing_values(small_xy):
    X, _ = small_xy
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_default_model_is_ridge():
    model = build_pipeline()
    assert isinstance(model.regressor.named_steps["model"], Ridge)


def test_predictions_are_in_dollars(small_xy):
    X, y = small_xy
    model = build_pipeline().fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    # log1p/expm1 wrapper must return dollars, not log values (~12)
    assert preds.min() > 10_000
    assert abs(np.median(preds) - y.median()) / y.median() < 0.2


def test_unknown_category_does_not_crash(small_xy):
    X, y = small_xy
    model = build_pipeline().fit(X, y)
    new_house = X.head(1).copy()
    new_house["Neighborhood"] = "NeverSeenBefore"
    assert model.predict(new_house)[0] > 0


def test_pipeline_accepts_tree_models(small_xy):
    """Dense preprocessor output, so models that reject sparse input still work."""
    X, y = small_xy
    model = build_pipeline(HistGradientBoostingRegressor(max_iter=10)).fit(X, y)
    assert len(model.predict(X.head(5))) == 5
