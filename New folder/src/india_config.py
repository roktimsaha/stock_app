"""
india_config.py
Auto-curates daily top performers by market-cap segment from NSE India.
Exposes:
- get_dynamic_tickers(top_large, top_mid, top_small)
- NSE_TICKERS: computed at import (uses cache & fallbacks)
- BUY_TH, SELL_TH, ROUND_TRIP_COST_BPS
"""

from __future__ import annotations
import os, json
from typing import Dict, List
from urllib.parse import quote
import requests

# User knobs
TOP_LARGE_DEFAULT = 10
TOP_MID_DEFAULT   = 10
TOP_SMALL_DEFAULT = 10

BUY_TH  = 0.55
SELL_TH = 0.45
ROUND_TRIP_COST_BPS = 15

INDEX_LARGE = "NIFTY 50"
INDEX_MID   = "NIFTY MIDCAP 100"
INDEX_SMALL = "NIFTY SMALLCAP 100"

ROOT_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
CACHE_PATH = os.path.join(DATA_DIR, "dynamic_tickers.json")

SEED_FALLBACK: Dict[str, List[str]] = {
    "LARGE": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
    "MID":   ["LTIM.NS", "DIXON.NS", "ABBOTINDIA.NS", "INDHOTEL.NS", "TATAELXSI.NS"],
    "SMALL": ["TEJASNET.NS", "MAPMYINDIA.NS", "KPITTECH.NS", "TANLA.NS", "COCHINSHIP.NS"],
}

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    })
    try:
        s.get("https://www.nseindia.com/", timeout=8)
    except Exception:
        pass
    return s

def _fetch_index(index_name: str, session: requests.Session) -> List[dict]:
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={quote(index_name)}"
    r = session.get(url, timeout=10)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("Unexpected NSE payload")
    return payload["data"]

def _top_gainers_symbols(index_name: str, top_n: int, session: requests.Session) -> List[str]:
    rows = _fetch_index(index_name, session)
    clean = []
    for row in rows:
        sym = row.get("symbol")
        pchg = row.get("pChange")
        if not sym:
            continue
        try:
            pchg_f = float(pchg)
        except Exception:
            continue
        clean.append((sym, pchg_f))
    clean.sort(key=lambda x: x[1], reverse=True)
    return [f"{sym}.NS" for sym, _ in clean[:max(0, int(top_n))]]

def _load_cache() -> Dict[str, List[str]] | None:
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_cache(payload: Dict[str, List[str]]) -> None:
    _ensure_data_dir()
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_dynamic_tickers(top_large: int = TOP_LARGE_DEFAULT,
                        top_mid:   int = TOP_MID_DEFAULT,
                        top_small: int = TOP_SMALL_DEFAULT) -> List[str]:
    try:
        s = _nse_session()
        large = _top_gainers_symbols(INDEX_LARGE, top_large, s)
        mid   = _top_gainers_symbols(INDEX_MID,   top_mid,   s)
        small = _top_gainers_symbols(INDEX_SMALL, top_small, s)
        combined, seen = [], set()
        for bucket in (large, mid, small):
            for t in bucket:
                if t not in seen:
                    seen.add(t)
                    combined.append(t)
        _save_cache({"LARGE": large, "MID": mid, "SMALL": small})
        return combined
    except Exception:
        cached = _load_cache()
        if cached:
            combined, seen = [], set()
            for bucket in (cached.get("LARGE", []), cached.get("MID", []), cached.get("SMALL", [])):
                for t in bucket:
                    if t not in seen:
                        seen.add(t)
                        combined.append(t)
            if combined:
                return combined
        combined, seen = [], set()
        for bucket in (SEED_FALLBACK["LARGE"], SEED_FALLBACK["MID"], SEED_FALLBACK["SMALL"]):
            for t in bucket:
                if t not in seen:
                    seen.add(t)
                    combined.append(t)
        return combined

NSE_TICKERS: List[str] = get_dynamic_tickers()

if __name__ == "__main__":
    print("Top tickers today:")
    for t in NSE_TICKERS:
        print("  -", t)
