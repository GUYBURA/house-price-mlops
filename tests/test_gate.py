import json

import joblib
import numpy as np
import pytest

from src import config
from src.gate import check_metrics, check_predictions, load_production_model, promote


def metrics(rmse_log):
    return {"rmse_log": rmse_log}


# --- check_predictions: sanity ---

def test_normal_predictions_pass():
    assert check_predictions(np.array([150_000.0, 200_000.0])) == []


def test_nan_predictions_fail():
    assert check_predictions(np.array([150_000.0, np.nan])) == ["1 predictions are NaN"]


def test_non_positive_predictions_fail():
    assert check_predictions(np.array([150_000.0, 0.0, -5.0])) == ["2 predictions are <= 0"]


# --- check_metrics: absolute bar + comparison ---

def test_first_run_passes_when_under_the_bar():
    assert check_metrics(metrics(0.094), production_metrics=None) == []


def test_first_run_fails_above_the_bar():
    """No production model must not mean 'anything goes'."""
    reasons = check_metrics(metrics(config.MAX_RMSE_LOG + 0.01), production_metrics=None)
    assert len(reasons) == 1 and "above the bar" in reasons[0]


def test_better_than_production_passes():
    assert check_metrics(metrics(0.090), metrics(0.094)) == []


def test_tiny_difference_within_tolerance_passes():
    """Test-score noise should not block every retrain."""
    within = 0.094 * (1 + config.GATE_TOLERANCE / 2)
    assert check_metrics(metrics(within), metrics(0.094)) == []


def test_clearly_worse_than_production_fails():
    reasons = check_metrics(metrics(0.110), metrics(0.094))
    assert len(reasons) == 1 and "worse than production" in reasons[0]


def test_worse_and_above_bar_reports_both():
    reasons = check_metrics(metrics(0.20), metrics(0.094))
    assert len(reasons) == 2


# --- production folder handling ---

def test_no_production_model_returns_none(tmp_path):
    assert load_production_model(tmp_path / "model.joblib") is None


def test_promote_copies_model_and_metrics(tmp_path):
    run_dir = tmp_path / "runs" / "abc123"
    run_dir.mkdir(parents=True)
    joblib.dump({"fake": "model"}, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps({"rmse_log": 0.09}), encoding="utf-8")
    production = tmp_path / "production"

    promote(run_dir / "model.joblib", run_dir / "metrics.json", production)

    assert load_production_model(production / "model.joblib") == {"fake": "model"}
    assert json.loads((production / "metrics.json").read_text(encoding="utf-8")) == {"rmse_log": 0.09}


def test_promote_replaces_old_production(tmp_path):
    production = tmp_path / "production"
    production.mkdir()
    joblib.dump("old", production / "model.joblib")
    (production / "metrics.json").write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    joblib.dump("new", run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    promote(run_dir / "model.joblib", run_dir / "metrics.json", production)

    assert load_production_model(production / "model.joblib") == "new"


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([-1.0])])
def test_sanity_failure_is_never_empty(bad):
    assert check_predictions(bad)
