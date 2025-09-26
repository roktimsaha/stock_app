"""
backtest.py (India aware)
Adds per-trade round-trip cost (bps). BUY when pred==1, hold 1 day.
"""

import os
import joblib
import pandas as pd
import numpy as np
from features import build_features_for_ticker
from train import time_train_test_split, prepare_Xy
from india_config import ROUND_TRIP_COST_BPS
import matplotlib.pyplot as plt

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def backtest(ticker="RELIANCE.NS", test_size=0.2):
    art_path = os.path.join(MODEL_DIR, f"{ticker}_rf_model.joblib")
    if not os.path.exists(art_path):
        raise RuntimeError(f"Model artifact not found for {ticker}. Train model first.")

    art = joblib.load(art_path)
    model = art["model"]
    features = art["features"]
    horizon = art.get("horizon", 1)

    df = build_features_for_ticker(ticker, horizon=horizon)
    train_df, test_df = time_train_test_split(df, test_size=test_size)
    X_test, y_test = prepare_Xy(test_df)
    X_test = X_test.reindex(columns=features).fillna(0)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    test_df = test_df.reset_index(drop=True)
    test_df["pred"] = preds
    test_df["prob"] = probs

    test_df["next_return"] = test_df["Close"].shift(-1) / test_df["Close"] - 1
    test_df = test_df.dropna().reset_index(drop=True)

    per_trade_cost = ROUND_TRIP_COST_BPS / 10000.0
    gross = test_df["pred"] * test_df["next_return"]
    net = np.where(test_df["pred"] == 1, gross - per_trade_cost, 0.0)

    test_df["strategy_ret"] = net
    test_df["market_ret"] = test_df["next_return"]
    test_df["strategy_equity"] = (1 + test_df["strategy_ret"]).cumprod()
    test_df["market_equity"] = (1 + test_df["market_ret"]).cumprod()

    total_strat = test_df["strategy_equity"].iloc[-1] - 1
    total_mkt = test_df["market_equity"].iloc[-1] - 1

    print(f"{ticker} | Strategy: {total_strat:.2%}  | Buy&Hold: {total_mkt:.2%}  | Cost(bps): {ROUND_TRIP_COST_BPS}")

    plt.figure(figsize=(8,5))
    plt.plot(test_df["date"], test_df["strategy_equity"], label="Strategy (net)")
    plt.plot(test_df["date"], test_df["market_equity"], label="Market")
    plt.legend(); plt.title(f"Equity curve ({ticker})")
    plt.xlabel("Date"); plt.ylabel("Equity"); plt.tight_layout()
    outpath = os.path.join(MODEL_DIR, f"{ticker}_backtest.png")
    plt.savefig(outpath)
    print("Saved equity plot at", outpath)

    return test_df

if __name__ == "__main__":
    backtest("RELIANCE.NS")
