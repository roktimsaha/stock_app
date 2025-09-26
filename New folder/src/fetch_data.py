# Source code

## `src/fetch_data.py`


"""
fetch_data.py
Fetches OHLCV data (Yahoo Finance) and saves into a local SQLite DB.
Works for NSE/BSE when you use suffixes: .NS (NSE), .BO (BSE).
"""

import os
from datetime import datetime
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stocks.db")
TABLE_NAME = "ohlcv"

def ensure_data_dir():
    d = os.path.dirname(DB_PATH)
    os.makedirs(d, exist_ok=True)

def download_to_sql(ticker: str, start: str="2015-01-01", end: str=None, interval: str="1d"):
    ensure_data_dir()
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(f"No data fetched for {ticker}")

    df.index = pd.to_datetime(df.index)
    df = df.rename_axis("date").reset_index()
    df["ticker"] = ticker

    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    df.to_sql(TABLE_NAME, con=engine, if_exists="append", index=False)
    print(f"Saved {len(df)} rows for {ticker} into {DB_PATH} (table {TABLE_NAME}).")

if __name__ == "__main__":
    # Example: fetch RELIANCE on NSE
    tickers = ["RELIANCE.NS"]
    for t in tickers:
        download_to_sql(t, start="2015-01-01")
