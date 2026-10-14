import joblib
import logging
from sklearn.linear_model import Ridge
from src.config import RANDOM_STATE, TEST_SIZE, MODEL_DIR, MODEL_PATH
from src.data import load_data, clean_data, remove_outliers
from src.features import add_features, split_target
from src.pipeline import build_pipeline
from src.evaluate import evaluate
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

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
    candidates = {
        "ridge": Ridge(alpha=10),
        "hgb": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "rf": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }
    for name, estimator in candidates.items():
        scores = cross_val_score(build_pipeline(estimator), X_train, y_train, cv=cv,
                             scoring="neg_root_mean_squared_error")
        logger.info(f"{name}: {-scores.mean():,.0f} +/- {scores.std():,.0f}")
    
    model = build_pipeline(candidates["ridge"]).fit(X_train, y_train)
    
    metrics = evaluate(y_test, model.predict(X_test))
    logger.info(
        "Test: MAE %.0f, RMSE %.0f, R2 %.3f, rmse_log %.3f",
        metrics["mae"], metrics["rmse (dollars)"], metrics["r2"], metrics["rmse_log"],
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)

if  __name__  ==  "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()