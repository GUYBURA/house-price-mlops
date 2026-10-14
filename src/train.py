import joblib
from src.config import RANDOM_STATE, TEST_SIZE, MODEL_DIR, MODEL_PATH
from src.data import load_data, clean_data, remove_outliers
from src.features import add_features, split_target
from src.pipeline import build_pipeline
from src.evaluate import evaluate
from sklearn.model_selection import train_test_split, cross_val_score, KFold


def main():
    data = load_data()
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
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(build_pipeline(), X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    print(f"CV RMSE: {-scores.mean():,.0f} ± {scores.std():,.0f}")
    
    model = build_pipeline().fit(X_train, y_train)
    
    metrics = evaluate(y_test, model.predict(X_test))
    print(metrics)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if  __name__  ==  "__main__":
    main()