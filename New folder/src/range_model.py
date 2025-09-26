"""
range_models.py
Quantile Gradient Boosting models for price RANGE forecasts (1,5,15-day).
Predict quantiles of future returns, convert to price (target/high/low).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.ensemble import GradientBoostingRegressor

from features import load_from_sql, add_technical_indicators

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_HORIZONS = [1, 5, 15]

def _build_regression_df(ticker: str) -> pd.DataFrame:
    df = load_from_sql(ticker)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df = add_technical_indicators(df)
    return df

def _make_supervised(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    out[f"ret_fwd_{horizon}"] = out["Close"].shift(-horizon) / out["Close"] - 1.0
    out = out.dropna().reset_index(drop=True)
    return out

def _prepare_Xy(df: pd.DataFrame, target_col: str):
    drop_cols = {"date","ticker","Adj Close", target_col}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df[target_col].astype(float).values
    return X, y

def _fit_quantile_gb(X, y, alpha: float, random_state: int = 42):
    gb = GradientBoostingRegressor(
        loss="quantile", alpha=alpha,
        n_estimators=350, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=random_state
    )
    gb.fit(X, y)
    return gb

def train_range_models_for_ticker(ticker: str, horizons: List[int] = None):
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    base = _build_regression_df(ticker)
    models: Dict[int, Dict[str, object]] = {}
    features_reference = None

    for h in horizons:
        sup = _make_supervised(base, horizon=h)
        if len(sup) < 250:
            continue
        X, y = _prepare_Xy(sup, target_col=f"ret_fwd_{h}")
        if features_reference is None:
            features_reference = list(X.columns)
        models[h] = {
            "q10": _fit_quantile_gb(X, y, alpha=0.10),
            "q50": _fit_quantile_gb(X, y, alpha=0.50),
            "q90": _fit_quantile_gb(X, y, alpha=0.90),
        }

    if not models:
        raise RuntimeError(f"No horizon trained for {ticker} (insufficient data).")

    artifact = {
        "ticker": ticker,
        "features": features_reference,
        "horizons": sorted(models.keys()),
        "models": models,
    }
    path = os.path.join(MODEL_DIR, f"{ticker}_ranges.joblib")
    joblib.dump(artifact, path)
    print(f"[range_models] Saved: {path}")
    return path

def _latest_features_row(ticker: str, feature_cols: List[str]):
    df = _build_regression_df(ticker)
    x = df.iloc[[-1]].drop(columns=[c for c in ["date","ticker","Adj Close"] if c in df.columns], errors="ignore")
    x = x.select_dtypes(include=[np.number]).reindex(columns=feature_cols, fill_value=0.0)
    last_close = float(df.iloc[-1]["Close"])
    last_date  = df.iloc[-1]["date"]
    return x, last_close, last_date

def predict_ranges_for_ticker(ticker: str):
    path = os.path.join(MODEL_DIR, f"{ticker}_ranges.joblib")
    if not os.path.exists(path):
        raise RuntimeError(f"Range artifact not found for {ticker}. Train ranges first.")

    art = joblib.load(path)
    feature_cols = art["features"]
    horizons     = art["horizons"]
    models       = art["models"]

    X, last_close, asof_date = _latest_features_row(ticker, feature_cols)

    out = {}
    for h in horizons:
        m = models[h]
        q10 = float(m["q10"].predict(X)[0])
        q50 = float(m["q50"].predict(X)[0])
        q90 = float(m["q90"].predict(X)[0])

        out[h] = {
            "return_low_q10": round(q10, 4),
            "return_med_q50": round(q50, 4),
            "return_high_q90": round(q90, 4),
            "price_low": round(last_close * (1.0 + q10), 2),
            "price_target": round(last_close * (1.0 + q50), 2),
            "price_high": round(last_close * (1.0 + q90), 2),
        }

    return {
        "ticker": ticker,
        "asof_date": asof_date,
        "last_close": last_close,
        "ranges": out
    }

if __name__ == "__main__":
    t = "RELIANCE.NS"
    train_range_models_for_ticker(t)
    print(predict_ranges_for_ticker(t))
