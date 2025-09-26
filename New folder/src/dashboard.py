"""
dashboard.py
Interactive Streamlit app for Indian stocks (NSE/BSE via yfinance).
Free-text single-ticker input + optional train + price range forecasts (1/5/15d).
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from sqlalchemy import create_engine
import joblib

from signal import latest_signal_for_ticker
from features import build_features_for_ticker
from train import time_train_test_split, prepare_Xy, train_and_save
from backtest import backtest as run_backtest_file
from india_config import BUY_TH, SELL_TH, ROUND_TRIP_COST_BPS, NSE_TICKERS
from range_models import predict_ranges_for_ticker, train_range_models_for_ticker

ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(ROOT, "data", "stocks.db")
MODEL_DIR = os.path.join(ROOT, "models")
TABLE_NAME = "ohlcv"

st.set_page_config(page_title="India Stock Predictor", layout="wide")
st.title("🇮🇳 Stock Price Movement Predictor — India (NSE/BSE)")
st.caption("Next-day direction model with price-range forecasts (1/5/15-day)")

def load_price_series(ticker: str):
    eng = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    q = f"SELECT date, Close FROM {TABLE_NAME} WHERE ticker='{ticker}' ORDER BY date"
    try:
        df = pd.read_sql(q, con=eng, parse_dates=["date"])
    except Exception:
        return pd.DataFrame(columns=["date","Close"])
    return df

def have_model(ticker: str):
    return os.path.exists(os.path.join(MODEL_DIR, f"{ticker}_rf_model.joblib"))

def have_range_artifact(ticker: str):
    return os.path.exists(os.path.join(MODEL_DIR, f"{ticker}_ranges.joblib"))

def fetch_append(ticker: str, start="2015-01-01", end=None, interval="1d"):
    import yfinance as yf
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    eng = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    st.write(f"Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError("No data fetched. Check the symbol (e.g., RELIANCE.NS).")
    df.index = pd.to_datetime(df.index)
    df = df.rename_axis("date").reset_index()
    df["ticker"] = ticker
    df.to_sql(TABLE_NAME, con=eng, if_exists="append", index=False)

def plot_price(px: pd.DataFrame, last_action: str, last_close: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px["date"], y=px["Close"], mode="lines", name="Close"))
    if not px.empty:
        fig.add_hline(y=last_close, line_dash="dot")
        fig.add_annotation(x=px["date"].iloc[-1], y=last_close, text=last_action, showarrow=True, arrowhead=2)
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
    return fig

tab1, tab2 = st.tabs(["Single Ticker (free input)", "Curated List (auto top performers)"])

with tab1:
    st.subheader("Single Ticker")
    ticker = st.text_input("Enter Yahoo Finance symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS, RELIANCE.BO)", "RELIANCE.NS").strip()

    colA, colB, colC = st.columns([1,1,1])
    start_date = colA.text_input("Start date for initial fetch (YYYY-MM-DD)", "2015-01-01")
    do_fetch   = colB.checkbox("Fetch/refresh data before training", value=True)
    do_train   = colC.checkbox("Train/refresh models for this ticker", value=True)

    run_btn = st.button("Load / Train / Show")
    if run_btn and ticker:
        if do_fetch:
            try:
                fetch_append(ticker, start=start_date)
                st.success("Data fetched & appended to SQLite.")
            except Exception as e:
                st.error(f"Fetch failed: {e}")

        try:
            if do_train or not have_model(ticker):
                train_and_save(ticker, horizon=1)
                st.success("Classification model trained.")
        except Exception as e:
            st.error(f"Training (classifier) failed: {e}")

        try:
            if do_train or not have_range_artifact(ticker):
                train_range_models_for_ticker(ticker)
                st.success("Range models (1/5/15-day) trained.")
        except Exception as e:
            st.error(f"Training (range models) failed: {e}")

        try:
            sig = latest_signal_for_ticker(ticker)
            if sig.get("status") == "ok":
                buy_th = st.slider("Buy threshold (prob up ≥)", 0.50, 0.80, float(BUY_TH), 0.01, key="buy_single")
                sell_th = st.slider("Sell threshold (prob up ≤)", 0.20, 0.50, float(SELL_TH), 0.01, key="sell_single")

                prob_up = sig["prob_up"]
                if prob_up >= buy_th:
                    action = "BUY"
                elif prob_up <= sell_th:
                    action = "SELL"
                else:
                    action = "HOLD"

                st.markdown(
                    f"### {ticker} — **{action}** (p(up)={prob_up:.2f})  |  Close ₹{sig['close']:.2f}  "
                    f"| as of {pd.to_datetime(sig['asof_date']).date()}"
                )

                px = load_price_series(ticker)
                st.plotly_chart(plot_price(px, action, sig["close"]), use_container_width=True)

                roc_p = os.path.join(MODEL_DIR, f"{ticker}_roc.png")
                eq_p  = os.path.join(MODEL_DIR, f"{ticker}_backtest.png")
                cols = st.columns(2)
                if os.path.exists(roc_p):
                    cols[0].image(roc_p, caption=f"{ticker} ROC", use_column_width=True)
                if os.path.exists(eq_p):
                    cols[1].image(eq_p, caption=f"{ticker} Equity", use_column_width=True)
            else:
                st.warning(f"Signal unavailable: {sig.get('status')}")
        except Exception as e:
            st.warning(f"Signal computation failed: {e}")

        st.markdown("**Price Range Forecasts**")
        try:
            rng = predict_ranges_for_ticker(ticker)
            r = rng["ranges"]
            rows = []
            for H in [1,5,15]:
                if H in r:
                    rows.append({
                        "Horizon": f"{H}-day",
                        "Target": r[H]["price_target"],
                        "High":   r[H]["price_high"],
                        "Low":    r[H]["price_low"],
                        "Ret q10": r[H]["return_low_q10"],
                        "Ret q50": r[H]["return_med_q50"],
                        "Ret q90": r[H]["return_high_q90"],
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("Range models not available; train them above.")
        except Exception as e:
            st.warning(f"Range prediction unavailable: {e}")

        st.subheader("Quick Backtest")
        cost_bps = st.number_input("Round-trip cost (bps)", min_value=0, max_value=200, value=int(ROUND_TRIP_COST_BPS), step=1, key="cost_single")
        test_size = st.slider("Test size (fraction)", 0.1, 0.4, 0.2, 0.01, key="test_single")
        if st.button("Run backtest (save & show)"):
            try:
                df_bt = run_backtest_file(ticker, test_size=float(test_size))
                st.success("Backtest completed. Equity curve saved in /models.")
                st.dataframe(df_bt[["date","Close","pred","prob","next_return","strategy_ret"]].tail(10), use_container_width=True)
            except Exception as e:
                st.error(f"Backtest failed: {e}")

with tab2:
    st.subheader("Curated List (Top performers by segment)")
    st.caption("Auto list comes from `india_config.py` each run.")
    tickers = st.multiselect("Select curated tickers", NSE_TICKERS, default=NSE_TICKERS[:6])
    buy_th2  = st.slider("Buy threshold (prob up ≥)", 0.50, 0.80, float(BUY_TH), 0.01, key="buy_cur")
    sell_th2 = st.slider("Sell threshold (prob up ≤)", 0.20, 0.50, float(SELL_TH), 0.01, key="sell_cur")

    if not tickers:
        st.info("Pick at least one ticker.")
    else:
        for t in tickers:
            try:
                sig = latest_signal_for_ticker(t)
                if sig.get("status") != "ok":
                    st.warning(f"{t}: {sig.get('status')}")
                    continue
                prob_up = sig["prob_up"]
                action = "BUY" if prob_up >= buy_th2 else ("SELL" if prob_up <= sell_th2 else "HOLD")
                st.markdown(f"### {t} — **{action}** (p(up)={prob_up:.2f})  |  Close ₹{sig['close']:.2f}  | as of {pd.to_datetime(sig['asof_date']).date()}")
                px = load_price_series(t)
                st.plotly_chart(plot_price(px, action, sig["close"]), use_container_width=True)

                try:
                    rng = predict_ranges_for_ticker(t)
                    r = rng["ranges"]
                    rows = []
                    for H in [1,5,15]:
                        if H in r:
                            rows.append({"Horizon": f"{H}-day", "Target": r[H]["price_target"], "High": r[H]["price_high"], "Low": r[H]["price_low"]})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception:
                    pass
            except Exception as e:
                st.warning(f"{t}: {e}")
