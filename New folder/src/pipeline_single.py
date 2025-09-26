"""
pipeline_single.py
One-shot pipeline for a single ticker:
fetch -> train classifier -> train range models -> backtest

Usage:
  python src/pipeline_single.py --ticker RELIANCE.NS --start 2015-01-01
"""

import argparse
import os
from datetime import datetime
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

from train import train_and_save
from backtest import backtest
from range_models import train_range_models_for_ticker

ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(ROOT, "data", "stocks.db")
TABLE_NAME = "ohlcv"

def ensure_dirs():
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

def fetch_to_sql(ticker: str, start="2015-01-01", end=None, interval="1d"):
    ensure_dirs()
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    eng = create_engine(f"sqlite:///{DB_PATH}", echo=False)

    print(f"[fetch] {ticker} {start}→{end} ({interval})")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(f"No data fetched for {ticker}. Check symbol or date range.")
    df.index = pd.to_datetime(df.index)
    df = df.rename_axis("date").reset_index()
    df["ticker"] = ticker
    df.to_sql(TABLE_NAME, con=eng, if_exists="append", index=False)
    print(f"[fetch] saved {len(df)} rows to {DB_PATH}:{TABLE_NAME}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True, help="Yahoo symbol (e.g., RELIANCE.NS, TCS.NS, RELIANCE.BO)")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()

    fetch_to_sql(args.ticker, start=args.start, end=args.end)
    train_and_save(args.ticker, horizon=1)
    train_range_models_for_ticker(args.ticker)
    backtest(args.ticker)

    print("\nDone. Open the Streamlit app to view: `streamlit run src/dashboard.py`")

if __name__ == "__main__":
    main()
