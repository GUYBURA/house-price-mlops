# Create Project

# Write order (each file depends on the ones before):
1. config.py: constants only, no logic
2. data.py: load_data(), clean(df)
3. features.py: add_features(df)
4. pipeline.py: build_pipeline()
5. evaluate.py: evaluate(y_true, y_pred)
6. train.py: ties it all together
7. predict.py: last