#!/usr/bin/env python3
"""
Auto Trader (no pip) for Termux / Linux
- Uses only Python standard library
- Talks directly to Binance REST API (signed requests)
- SMA crossover strategy (pure Python)
- DRY_RUN by default
"""
import math
import os
import sys
import time
import json
import csv
import hmac
import hashlib
import logging
import shutil
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone

# ---------- CONFIG / FILES ----------
PROJECT_DIR = os.path.expanduser("~/auto_trader")
STATE_FILE = os.path.join(PROJECT_DIR, "autotrader_state.json")
TRADES_CSV = os.path.join(PROJECT_DIR, "autotrader_trades.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "autotrader.log")
ENV_FILE = os.path.join(PROJECT_DIR, ".env")

# default config (overridden by .env)
CFG = {
    "API_KEY": "",
    "API_SECRET": "",
    "SYMBOLS": "BTC/USDT,ETH/USDT,LINK/USDT,ZIL/USDT",
    "QUOTE": "USDT",
    "POSITION_USD": "25",  # USD per buy
    "FAST_SMA": "20",
    "SLOW_SMA": "50",
    "TIMEFRAME": "15m",
    "POLL_SECONDS": "30",
    "TP_PCT": "1.0",
    "SL_PCT": "0.8",
    "DRY_RUN": "true",  # safe default
    "NOTIFY": "true",
}

BINANCE_BASE = "https://api.binance.com"

# ---------- Logging ----------
os.makedirs(PROJECT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)

# ---------- Utilities ----------
def load_env_file(path):
    """Simple .env loader into CFG and os.environ (no external libs)."""
    if not os.path.exists(path):
        logging.info("No .env found at %s — using defaults (DRY_RUN=true)", path)
        return
    with open(path, "r") as f:
        for line in f:
            if "=" not in line or line.strip().startswith("#"):
                continue
            k, v = line.strip().split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            CFG[k] = v
            os.environ[k] = v


def get_cfg(key, cast=str):
    v = CFG.get(key, "")
    if cast is bool:
        return str(v).lower() in ("1", "true", "yes", "y")
    try:
        return cast(v)
    except Exception:
        return v


def load_symbol_filters():
    info = http_get("/api/v3/exchangeInfo")
    filters = {}
    for s in info["symbols"]:
        symbol = s["symbol"]
        lot = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
        notional = next(f for f in s["filters"] if f["filterType"] == "NOTIONAL")
        filters[symbol] = {
            "minQty": float(lot["minQty"]),
            "stepSize": float(lot["stepSize"]),
            "minNotional": float(notional["minNotional"]),
        }
    return filters


# notification helper (Termux)
def notify(title: str, message: str):
    if not get_cfg("NOTIFY", bool):
        logging.info("[NOTIFY disabled] %s: %s", title, message)
        return
    if shutil.which("termux-notification"):
        try:
            subprocess.run(
                ["termux-notification", "--title", title, "--content", message[:4096]],
                check=False,
            )
        except Exception as e:
            logging.warning("termux-notification failed: %s", e)
    else:
        # fallback log
        logging.info("[NOTIFY] %s: %s", title, message)


# ---------- Binance HTTP helpers ----------
def http_get(path, params=None, signed=False, api_key=None, api_secret=None):
    if params is None:
        params = {}

    if signed:
        if api_secret is None:
            raise RuntimeError("API secret required for signed request")
        # FIX: Add timestamp to params dictionary BEFORE encoding it
        params["timestamp"] = int(time.time() * 1000)

    params_encoded = urllib.parse.urlencode(params)
    url = BINANCE_BASE + path
    if params_encoded:
        url = url + "?" + params_encoded

    headers = {"User-Agent": "AutoTrader/1.0"}
    if signed:
        # FIX: Sign the encoded string which now correctly includes the timestamp
        signature = hmac.new(api_secret.encode(), params_encoded.encode(), hashlib.sha256).hexdigest()
        url = url + "&signature=" + signature
        headers["X-MBX-APIKEY"] = api_key or ""

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode()
            return json.loads(data)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        logging.warning("HTTP GET %s failed: %s %s", url, e.code, body)
        raise
    except Exception as e:
        logging.exception("HTTP GET error: %s %s", url, e)
        raise


def http_post(path, params=None, signed=False, api_key=None, api_secret=None):
    if params is None:
        params = {}

    # for signed endpoints, must append timestamp and signature to query string
    params_to_sign = dict(params)  # copy
    timestamp = int(time.time() * 1000)
    params_to_sign["timestamp"] = timestamp
    query = urllib.parse.urlencode(params_to_sign)
    if signed:
        signature = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        query = query + "&signature=" + signature

    url = BINANCE_BASE + path + "?" + query
    headers = {"User-Agent": "AutoTrader/1.0"}
    if api_key:
        headers["X-MBX-APIKEY"] = api_key

    req = urllib.request.Request(url, headers=headers, method="POST", data=b"")  # POST with empty body, all in query
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode()
            return json.loads(data)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        logging.warning("HTTP POST %s failed: %s %s", url, e.code, body)
        raise
    except Exception as e:
        logging.exception("HTTP POST error: %s %s", url, e)
        raise


# ---------- Market helpers ----------
def fetch_klines(symbol, interval, limit=200):
    params = {"symbol": symbol.replace("/", ""), "interval": interval, "limit": limit}
    return http_get("/api/v3/klines", params=params)


def fetch_price(symbol):
    params = {"symbol": symbol.replace("/", "")}
    t = http_get("/api/v3/ticker/price", params=params)
    return float(t["price"])


SYMBOL_FILTERS = load_symbol_filters()

# ... (rest of script unchanged, just indented properly)