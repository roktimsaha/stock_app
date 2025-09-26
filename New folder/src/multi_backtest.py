"""
multi_backtest.py
Runs the simple backtest for a list of tickers. Saves equity curves & summary CSV.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from features import build_features_for_ticker
from train import time_train_test_split, prepare_Xy
from india_config import ROUND_TRIP_COST_BPS, NSE_TICKERS

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def backtest_ticker(ticker: str, test_size=0.2):
    art_path = os.path.join(MODEL_DIR, f"{ticker}_rf_model.joblib")
    if not os.path.exists(art_path):
        raise RuntimeError(f"Model artifact missing for {ticker}. Train first.")

    artifact = joblib.load(art_path)
    model, features = artifact["model"], artifact["features"]
    df = build_features_for_ticker(ticker, horizon=1)
    tr, te = time_train_test_split(df, test_size=test_size)
    X_test, y_test = prepare_Xy(te)
    X_test = X_test.reindex(columns=features).fillna(0)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    te = te.reset_index(drop=True)
    te["pred"] = preds
    te["prob"] = probs
    te["next_return"] = te["Close"].shift(-1) / te["Close"] - 1
    te = te.dropna().reset_index(drop=True)

    per_trade_cost = ROUND_TRIP_COST_BPS / 10000.0
    gross = te["pred"] * te["next_return"]
    net = np.where(te["pred"] == 1, gross - per_trade_cost, 0.0)

    te["strategy_ret"] = net
    te["market_ret"] = te["next_return"]
    te["strategy_equity"] = (1 + te["strategy_ret"]).cumprod()
    te["market_equity"] = (1 + te["market_ret"]).cumprod()

    total_strat = te["strategy_equity"].iloc[-1] - 1
    total_mkt = te["market_equity"].iloc[-1] - 1

    plt.figure(figsize=(8, 5))
    plt.plot(te["date"], te["strategy_equity"], label="Strategy (net)")
    plt.plot(te["date"], te["market_equity"], label="Market")
    plt.title(f"Equity curve - {ticker}")
    plt.xlabel("Date"); plt.ylabel("Equity"); plt.legend(); plt.tight_layout()
    out_png = os.path.join(MODEL_DIR, f"{ticker}_backtest.png")
    plt.savefig(out_png); plt.close()

    return {
        "ticker": ticker,
        "strategy_total_return": round(float(total_strat), 4),
        "market_total_return": round(float(total_mkt), 4),
        "equity_plot": os.path.basename(out_png),
    }

def main():
    rows = []
    for t in NSE_TICKERS:
        try:
            rows.append(backtest_ticker(t))
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    if rows:
        out_csv = os.path.join(MODEL_DIR, "backtest_summary.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("Saved:", out_csv)

if __name__ == "__main__":
    main()
