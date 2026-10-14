# Architecture
- scikit-learn  (training model)
- pandas    (data cleaning)
- matplotlib (data visualize)
- seaborn   (data visualize)
- jupyter   (Display and tested on local)
- joblib (Saving model)
- numpy (Math)

# Folder Structure

├── data        (AmesHousing.csv)
├── docs        (project documents)
├── models      (saved model.joblib, not in git)
├── notebook    (EDA, local testing)
└── src
    ├── config.py    (paths, column lists, constants)
    ├── data.py      (load, clean, remove outliers)
    ├── features.py  (new features, drop columns, split target)
    ├── pipeline.py  (imputers, scaler, one-hot, model, log target)
    ├── evaluate.py  (RMSE, MAE, R2)
    ├── train.py     (compare models, train, test, save)
    └── predict.py   (load model, price new houses)

# Data
- Ames Housing Dataset (Because it has a lot of column to use for learning)

# Flow
    load_data -> clean_data -> remove_outliers (train only) -> add_features -> split -> pipeline -> evaluate -> save

# Evaluate
- RMSE (on log price): main metric, also the Kaggle metric for Ames
- MAE: average error in dollars, easy to explain to non-technical people
- R²: how much of the price variation the model explains

# Model
- Ridge (alpha=10) on log1p (SalePrice), via TransformedTargetRegressor
- Chosen by 5-fold CV RMSE on train: Ridge 20489 / HistGradientBoosting 22977 / RandomForest 25673
- Test: RMSE $18,260, MAE $12,535, R2 0.959, rmse_log 0.094
- Why Ridge: small data, and the target is transform with log1p and it make more linear so Ridge is a linear model so it make a better result.