import logging
import shutil
from pathlib import Path

import joblib
import numpy as np

from src import config

logger = logging.getLogger(__name__)


def check_predictions(y_pred) -> list[str]:
    """Sanity checks. Metrics are meaningless (or crash) if these fail, so run them first."""
    y_pred = np.asarray(y_pred, dtype=float)
    reasons = []
    n_nan = int(np.isnan(y_pred).sum())
    if n_nan:
        reasons.append(f"{n_nan} predictions are NaN")
    n_non_positive = int((y_pred[~np.isnan(y_pred)] <= 0).sum())
    if n_non_positive:
        reasons.append(f"{n_non_positive} predictions are <= 0")
    return reasons


def check_metrics(new_metrics: dict, production_metrics: dict | None) -> list[str]:
    """Absolute bar, then comparison with production scored on the SAME test set."""
    reasons = []
    new = new_metrics["rmse_log"]
    if new > config.MAX_RMSE_LOG:
        reasons.append(f"rmse_log {new:.4f} above the bar {config.MAX_RMSE_LOG}")

    if production_metrics is not None:
        allowed = production_metrics["rmse_log"] * (1 + config.GATE_TOLERANCE)
        if new > allowed:
            reasons.append(
                f"rmse_log {new:.4f} worse than production {production_metrics['rmse_log']:.4f} "
                f"(allowed up to {allowed:.4f})"
            )
    return reasons


def load_production_model(path: Path = config.PRODUCTION_MODEL_PATH):
    """Current approved model, or None on the first run."""
    if not path.exists():
        return None
    return joblib.load(path)


def promote(model_path: Path, metrics_path: Path, production_dir: Path = config.PRODUCTION_DIR) -> None:
    """Copy the candidate's model and metrics into the production folder."""
    production_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, production_dir / model_path.name)
    shutil.copy2(metrics_path, production_dir / metrics_path.name)
    logger.info("Promoted model to %s", production_dir)
