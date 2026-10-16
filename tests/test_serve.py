import json

import joblib
import pytest
from fastapi.testclient import TestClient

from src import config, serve
from src.pipeline import build_pipeline


def to_instances(df):
    """What a real client sends: JSON, so NaN becomes null."""
    return json.loads(df.to_json(orient="records"))


@pytest.fixture
def model_dir(small_xy, tmp_path):
    X, y = small_xy
    joblib.dump(build_pipeline().fit(X, y), tmp_path / "model.joblib")
    return tmp_path


@pytest.fixture
def client(model_dir, monkeypatch):
    """Server started like Vertex starts it: AIP_STORAGE_URI points at the model folder."""
    monkeypatch.setenv("AIP_STORAGE_URI", str(model_dir))
    with TestClient(serve.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_returns_one_price_per_house(client, raw):
    houses = raw.drop(columns=[config.TARGET, "Order", "PID"]).head(3)
    response = client.post("/predict", json={"instances": to_instances(houses)})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert all(p > 10_000 for p in predictions)


def test_single_house_with_nulls_works(client, raw):
    """One house: columns like Alley are all null, the hardest case for dtypes."""
    house = raw.drop(columns=[config.TARGET, "Order", "PID"]).head(1)
    response = client.post("/predict", json={"instances": to_instances(house)})
    assert response.status_code == 200


def test_server_matches_predict_script(client, raw, model_dir):
    """Train/serve consistency: HTTP result equals the shared predict path."""
    from src.predict import predict_with_model

    houses = raw.head(5)
    expected = predict_with_model(joblib.load(model_dir / "model.joblib"), houses)
    response = client.post("/predict", json={"instances": to_instances(houses)})
    assert response.json()["predictions"] == pytest.approx(list(expected))


def test_invalid_data_returns_400_with_reason(client, raw):
    house = raw.drop(columns=config.TARGET).head(1).copy()
    house["Kitchen Qual"] = "Excellent"
    response = client.post("/predict", json={"instances": to_instances(house)})
    assert response.status_code == 400
    assert "Kitchen Qual" in response.json()["detail"]


def test_missing_column_returns_400(client, raw):
    house = raw.drop(columns=[config.TARGET, "Year Built"]).head(1)
    response = client.post("/predict", json={"instances": to_instances(house)})
    assert response.status_code == 400
    assert "Year Built" in response.json()["detail"]


def test_empty_instances_returns_400(client):
    assert client.post("/predict", json={"instances": []}).status_code == 400


def test_wrong_request_shape_returns_422(client):
    """Missing 'instances' key: FastAPI rejects it before our code runs."""
    assert client.post("/predict", json={"rows": []}).status_code == 422


def test_resolve_local_folder(model_dir):
    assert serve.resolve_model_path(str(model_dir)) == model_dir / "model.joblib"


def test_resolve_defaults_to_production():
    assert serve.resolve_model_path(None) == config.PRODUCTION_MODEL_PATH
