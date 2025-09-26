"""
train.py
Trains a classification model to predict next-day up/down. Saves model + ROC plot.
"""

import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from features import build_features_for_ticker

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def time_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    n = len(df)
    split = int(n * (1 - test_size))
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test

def prepare_Xy(df: pd.DataFrame):
    drop_cols = ["date","ticker","target","Adj Close"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df["target"].astype(int)
    return X, y

def train_and_save(ticker: str="RELIANCE.NS", horizon: int=1):
    df = build_features_for_ticker(ticker, horizon=horizon)
    train_df, test_df = time_train_test_split(df, test_size=0.2)
    X_train, y_train = prepare_Xy(train_df)
    X_test, y_test = prepare_Xy(test_df)

    print("Train rows:", len(X_train), "Test rows:", len(X_test))
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:,1]
    print("Logistic Regression AUC:", roc_auc_score(y_test, lr_probs))

    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    param_grid = {"n_estimators": [150, 300], "max_depth": [6, 12, None]}
    gs = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    preds = best.predict(X_test)
    probs = best.predict_proba(X_test)[:,1]

    print("Random Forest")
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))
    print("Accuracy:", accuracy_score(y_test, preds))

    artifact = {"model": best, "features": list(X_train.columns), "ticker": ticker, "horizon": horizon}
    path = os.path.join(MODEL_DIR, f"{ticker}_rf_model.joblib")
    joblib.dump(artifact, path)
    print("Saved model to", path)

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"RF AUC={roc_auc_score(y_test, probs):.3f}")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {ticker}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{ticker}_roc.png"))
    print("Saved ROC plot.")

if __name__ == "__main__":
    train_and_save("RELIANCE.NS", horizon=1)
