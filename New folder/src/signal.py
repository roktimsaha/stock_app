"""
signal.py
Load saved model and compute latest Buy/Sell/Hold signal for a ticker.
"""

import os
import joblib
import numpy as np
import pandas as pd
from features import build_features_for_ticker
from india_config import BUY_TH, SELL_TH

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def latest_signal_for_ticker(ticker: str):
    art_path = os.path.join(MODEL_DIR, f"{ticker}_rf_model.joblib")
    if not os.path.exists(art_path):
        return {"ticker": ticker, "status": "no_model"}

    art = joblib.load(art_path)
    model, features = art["model"], art["features"]
    horizon = art.get("horizon", 1)

    df = build_features_for_ticker(ticker, horizon=horizon)
    if df.empty:
        return {"ticker": ticker, "status": "no_data"}

    X = df.iloc[[-1]].drop(columns=[c for c in ["date","ticker","target","Adj Close"] if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).reindex(columns=features, fill_value=0)

    prob_up = float(model.predict_proba(X)[0,1])
    pred = int(prob_up >= 0.5)
    if prob_up >= BUY_TH:
        action = "BUY"
    elif prob_up <= SELL_TH:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "ticker": ticker,
        "status": "ok",
        "prob_up": round(prob_up, 4),
        "pred": pred,
        "action": action,
        "horizon": horizon,
        "asof_date": df.iloc[-1]["date"],
        "close": float(df.iloc[-1]["Close"])
    }
