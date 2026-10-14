import joblib
import pytest

from src import config
from src import predict as predict_module
from src.pipeline import build_pipeline


@pytest.fixture
def saved_model(small_xy, tmp_path, monkeypatch):
    """Train a small model into a temp folder, so tests never touch models/model.joblib."""
    X, y = small_xy
    path = tmp_path / "model.joblib"
    joblib.dump(build_pipeline().fit(X, y), path)
    monkeypatch.setattr(predict_module, "MODEL_PATH", path)
    return path


def test_predict_raw_rows_with_target(saved_model, raw):
    preds = predict_module.predict(raw.head(5))
    assert len(preds) == 5
    assert (preds > 10_000).all()


def test_predict_new_house_without_target_or_ids(saved_model, raw):
    """A real new house: no SalePrice, no Order/PID."""
    new_houses = raw.drop(columns=[config.TARGET, "Order", "PID"]).head(3)
    assert len(predict_module.predict(new_houses)) == 3


def test_predict_matches_training_preparation(saved_model, raw, small_xy):
    """Train/serve consistency: predict() on raw rows must equal model.predict on prepared rows."""
    X, _ = small_xy
    model = joblib.load(saved_model)
    from_raw = predict_module.predict(raw.head(5))
    from_prepared = model.predict(X.head(5))
    assert from_raw == pytest.approx(from_prepared)


def test_predict_never_drops_houses(saved_model, raw):
    """Outlier removal is training-only; every house asked about gets a price."""
    big_houses = raw[raw["Gr Liv Area"] > 4000]
    assert len(predict_module.predict(big_houses)) == len(big_houses)
