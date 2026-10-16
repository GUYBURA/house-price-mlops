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
from src.gate import check_metrics, check_predictions, load_production_model, promote
from src.pipeline import build_pipeline
from src.validate import validate

logger = logging.getLogger(__name__)

def main():
    data = load_data()
    logger.info("Loaded %d rows", len(data))
    validate(data)  # before cleaning: cleaning would hide broken data
    logger.info("Data validation passed")
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

    y_pred = model.predict(X_test)
    gate_reasons = check_predictions(y_pred)
    metrics = evaluate(y_test, y_pred) if not gate_reasons else {}
    if metrics:
        logger.info(
            "Test: MAE %.0f, RMSE %.0f, R2 %.3f, rmse_log %.3f",
            metrics["mae"], metrics["rmse_dollars"], metrics["r2"], metrics["rmse_log"],
        )

    # Re-score production on THIS test set; its old metrics.json used a different one
    production_metrics = None
    production_model = load_production_model()
    if production_model is None:
        logger.info("No production model yet: gate checks the absolute bar only")
    else:
        try:
            production_metrics = evaluate(y_test, production_model.predict(X_test))
            logger.info("Production on same test set: rmse_log %.4f", production_metrics["rmse_log"])
        except Exception as error:  # e.g. features changed since production was trained
            logger.warning("Cannot score production model on new data (%s): absolute bar only", error)

    if not gate_reasons:
        gate_reasons = check_metrics(metrics, production_metrics)

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
        "run_dir": str(MODEL_DIR),
        "gate": {
            "passed": not gate_reasons,
            "reasons": gate_reasons,
            "production_test": (
                {k: round(float(v), 4) for k, v in production_metrics.items()} if production_metrics else None
            ),
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved metrics to %s", METRICS_PATH)

    # Candidate is always saved above for inspection; production changes only on a pass
    if gate_reasons:
        logger.warning("Quality gate FAILED, production unchanged: %s", "; ".join(gate_reasons))
    else:
        logger.info("Quality gate passed")
        promote(MODEL_PATH, METRICS_PATH)


if  __name__  ==  "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
