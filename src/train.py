import json
import logging
from datetime import UTC, datetime

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.config import METRICS_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.data import clean_data, load_data, remove_outliers
from src.evaluate import evaluate
from src.features import add_features, split_target
from src.pipeline import build_pipeline

logger = logging.getLogger(__name__)

def main():
    data = load_data()
    logger.info("Loaded %d rows", len(data))
    cleaned_data = clean_data(data)
    removed_outlier = remove_outliers(cleaned_data)
    features = add_features(removed_outlier)
    X, y = split_target(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    logger.info("Train %d rows, test %d rows", len(X_train), len(X_test))
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    candidates = {
        "ridge": Ridge(alpha=10),
        "hgb": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "rf": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }
    for name, estimator in candidates.items():
        scores = cross_val_score(build_pipeline(estimator), X_train, y_train, cv=cv,
                             scoring="neg_root_mean_squared_error")
        cv_results[name] = -scores.mean()
        logger.info(f"{name}: {-scores.mean():,.0f} +/- {scores.std():,.0f}")

    chosen = "ridge"
    estimator = candidates[chosen]

    model = build_pipeline(estimator).fit(X_train, y_train)

    metrics = evaluate(y_test, model.predict(X_test))
    logger.info(
        "Test: MAE %.0f, RMSE %.0f, R2 %.3f, rmse_log %.3f",
        metrics["mae"], metrics["rmse_dollars"], metrics["r2"], metrics["rmse_log"],
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)
    report = {
        "trained_at": datetime.now(UTC).isoformat(),
        "model": chosen,
        "params": estimator.get_params(),
        "data": {"rows_train": len(X_train), "rows_test": len(X_test)},
        "cv_rmse": {name: round(float(score), 1) for name, score in cv_results.items()},
        "test": {k: round(float(v), 4) for k, v in metrics.items()},
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved metrics to %s", METRICS_PATH)


if  __name__  ==  "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
