"""FastAPI inference endpoint and local demo page."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from iris_flow.constants import FEATURES, SPECIES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "iris_model.joblib"

app = FastAPI(title="Iris Flow API", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


class FlowerMeasurements(BaseModel):
    sepal_length: float = Field(ge=0.1, le=10, description="Centimeters")
    sepal_width: float = Field(ge=0.1, le=10, description="Centimeters")
    petal_length: float = Field(ge=0.1, le=10, description="Centimeters")
    petal_width: float = Field(ge=0.1, le=10, description="Centimeters")


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is missing. Run: dvc repro")
    return joblib.load(MODEL_PATH)


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict")
async def predict(measurements: FlowerMeasurements):
    model = get_model()
    row = pd.DataFrame([[getattr(measurements, key) for key in FEATURES]], columns=FEATURES)
    probabilities = model.predict_proba(row)[0]
    scores = {SPECIES[int(label)]: round(float(score), 4)
              for label, score in zip(model.classes_, probabilities)}
    winner = SPECIES[int(model.classes_[probabilities.argmax()])]
    return {"species": winner, "confidence": scores[winner], "probabilities": scores}

