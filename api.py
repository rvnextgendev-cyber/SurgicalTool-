"""
FastAPI service to expose the surgical tool usage predictor.
Start with: uvicorn api:app --reload --port 8000
"""

from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    operation_type: str = Field(..., example="Appendectomy")
    tool_name: str = Field(..., example="Scalpel")
    surgery_duration_min: int = Field(..., ge=1, le=1000, example=90)
    complexity_score: int = Field(..., ge=1, le=5, example=3)
    surgeon_experience_years: int = Field(..., ge=0, le=60, example=10)


class PredictResponse(BaseModel):
    predicted_usage: int
    raw_prediction: float


def _load_pipeline():
    artifacts: Dict[str, object] = joblib.load("model.pkl")
    return artifacts["pipeline"]


pipeline = _load_pipeline()
app = FastAPI(title="Surgical Tool Usage Predictor", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    raw_pred = float(pipeline.predict(df)[0])
    rounded = max(1, int(round(raw_pred)))
    return {"predicted_usage": rounded, "raw_prediction": raw_pred}
