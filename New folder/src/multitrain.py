"""
multi_train.py
Train per-ticker models end-to-end across NSE_TICKERS and save artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

from features import build_features_for_ticker
from train import time_train_test_split, prepare_Xy
from india_config import NSE_TICKERS

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_single(ticker: str, horizon: int = 1, test_size: float = 0.2):
    df = build_features_for_ticker(ticker, horizon=horizon)
    tr, te = time_train_test_split(df, test_size=test_size)
    X_train, y_train = prepare_Xy(tr)
    X_test, y_test   = prepare_Xy(te)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    param_grid = {"n_estimators": [150, 300], "max_depth": [6, 12, None]}
    gs = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    preds = best.predict(X_test)
    probs = best.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)

    artifact = {"model": best, "features": list(X_train.columns), "ticker": ticker, "horizon": horizon}
    path = os.path.join(MODEL_DIR, f"{ticker}_rf_model.joblib")
    joblib.dump(artifact, path)

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"RF AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title(f"ROC - {ticker}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{ticker}_roc.png"))
    plt.close()

    return {"ticker": ticker, "auc_rf": round(float(auc), 4), "acc_rf": round(float(acc), 4), "auc_lr": round(float(lr_auc), 4)}

def main():
    rows = []
    for t in NSE_TICKERS:
        try:
            print(f"=== {t} ===")
            rows.append(train_single(t))
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    if rows:
        out_csv = os.path.join(MODEL_DIR, "metrics.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("Saved metrics:", out_csv)

if __name__ == "__main__":
    main()
