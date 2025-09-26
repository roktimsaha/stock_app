"""
api.py
Simple FastAPI server: /predict (single row) and /range_predict/<ticker>.
"""

import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from range_models import predict_ranges_for_ticker

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
artifact_path = os.path.join(MODEL_DIR, "RELIANCE.NS_rf_model.joblib")  # change if needed

app = FastAPI(title="Stock Movement Predictor")

class Row(BaseModel):
    date: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float

@app.on_event("startup")
def load_model():
    global artifact
    if not os.path.exists(artifact_path):
        raise RuntimeError("Model not found; train first.")
    artifact = joblib.load(artifact_path)
    print("Model loaded.")

@app.post("/predict")
def predict(row: Row):
    try:
        payload = pd.DataFrame([row.dict()])
        features = artifact["features"]
        X = payload.select_dtypes(include=[np.number]).reindex(columns=features, fill_value=0)
        pred = artifact["model"].predict(X)[0]
        prob = artifact["model"].predict_proba(X)[0,1]
        return {"prediction": int(pred), "probability": float(prob)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/range_predict/{ticker}")
def range_predict(ticker: str):
    try:
        out = predict_ranges_for_ticker(ticker)
        fmt = {}
        for H in [1,5,15]:
            if H in out["ranges"]:
                row = out["ranges"][H]
                fmt[f"{H}day"] = {
                    "target": row["price_target"],
                    "high":   row["price_high"],
                    "low":    row["price_low"]
                }
        return {
            "ticker": out["ticker"],
            "asof_date": str(out["asof_date"]),
            "last_close": out["last_close"],
            "predictions": fmt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

