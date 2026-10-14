import numpy as np
import pytest

from src.evaluate import evaluate


def test_evaluate_matches_hand_calculation():
    metrics = evaluate([100000, 200000], [110000, 190000])
    assert metrics["mae"] == pytest.approx(10000)
    assert metrics["r2"] == pytest.approx(0.96)
    expected_rmse_log = np.sqrt(np.mean([
        (np.log1p(100000) - np.log1p(110000)) ** 2,
        (np.log1p(200000) - np.log1p(190000)) ** 2,
    ]))
    assert metrics["rmse_log"] == pytest.approx(expected_rmse_log)


def test_perfect_predictions_score_perfectly():
    y = [100000, 150000, 200000]
    metrics = evaluate(y, y)
    assert metrics["mae"] == 0
    assert metrics["rmse_log"] == 0
    assert metrics["r2"] == 1
