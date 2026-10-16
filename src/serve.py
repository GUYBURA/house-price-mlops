"""Prediction server following Vertex AI's custom container contract.

Vertex sends POST {"instances": [{column: value, ...}, ...]} to the predict route
and expects {"predictions": [...]} back. It calls the health route to check the
server is ready, and sets AIP_STORAGE_URI to the registered model's artifact folder.
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config
from src.predict import predict_with_model
from src.validate import DataValidationError

logger = logging.getLogger(__name__)
MODEL_FILE = "model.joblib"


class PredictRequest(BaseModel):
    instances: list[dict]


class PredictResponse(BaseModel):
    predictions: list[float]


def resolve_model_path(storage_uri: str | None) -> Path:
    """Vertex gives a gs:// folder (no /gcs mount in serving), locally a plain folder."""
    if not storage_uri:
        return config.PRODUCTION_MODEL_PATH
    if storage_uri.startswith("gs://"):
        return download_from_gcs(storage_uri)
    return Path(storage_uri) / MODEL_FILE


def download_from_gcs(storage_uri: str) -> Path:
    from google.cloud import storage  # only needed on Vertex

    bucket_name, _, prefix = storage_uri.removeprefix("gs://").partition("/")
    blob_name = f"{prefix.rstrip('/')}/{MODEL_FILE}" if prefix else MODEL_FILE
    local_path = Path(tempfile.mkdtemp()) / MODEL_FILE
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    logger.info("Downloaded gs://%s/%s", bucket_name, blob_name)
    return local_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # this module is the server entrypoint, so it configures logging (uvicorn only sets up its own)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # load once at startup, not per request
    path = resolve_model_path(os.environ.get("AIP_STORAGE_URI"))
    app.state.model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(status_code=400, detail="instances is empty")
    df = pd.DataFrame(request.instances)
    try:
        preds = predict_with_model(app.state.model, df)
    except DataValidationError as error:
        # bad input is the caller's fault: 400 with the exact problems, not a 500
        raise HTTPException(status_code=400, detail=str(error)) from error
    return PredictResponse(predictions=[float(p) for p in preds])
