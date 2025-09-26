"""
features.py
Loads OHLCV from SQLite, computes technical indicators & target label.
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import ta  # technical analysis library

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stocks.db")
TABLE_NAME = "ohlcv"

def load_from_sql(ticker: str):
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    query = f"SELECT * FROM {TABLE_NAME} WHERE ticker = '{ticker}' ORDER BY date ASC"
    df = pd.read_sql(query, con=engine, parse_dates=["date"])
    if not df.empty:
        df = df.drop_duplicates(subset=["ticker","date"]).sort_values("date")
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["Close"]

    # SMAs
    df["sma_5"] = close.rolling(5).mean()
    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()

    # EMAs
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # RSI
    df["rsi_14"] = ta.momentum.rsi(close, window=14)

    # MACD
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Bollinger
    bb = ta.volatility.BollingerBands(close)
    df["bb_h"] = bb.bollinger_hband()
    df["bb_l"] = bb.bollinger_lband()
    df["bb_percent"] = (close - df["bb_l"]) / (df["bb_h"] - df["bb_l"])

    # Returns & vol
    df["returns_1"] = close.pct_change(1)
    df["returns_5"] = close.pct_change(5)
    df["vol_5"] = df["returns_1"].rolling(5).std()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)
    return df

def add_target(df: pd.DataFrame, horizon: int = 1):
    df = df.copy()
    df["target"] = (df["Close"].shift(-horizon) > df["Close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df

def build_features_for_ticker(ticker: str, horizon: int = 1):
    df = load_from_sql(ticker)
    if df.empty:
        raise RuntimeError("No data for ticker")
    df = add_technical_indicators(df)
    df = add_target(df, horizon=horizon)
    return df

if __name__ == "__main__":
    df = build_features_for_ticker("RELIANCE.NS")
    print(df.head())
    print("Rows:", len(df))
