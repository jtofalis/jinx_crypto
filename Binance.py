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
import logging
# ---------- CONFIG / FILES ----------
PROJECT_DIR = os.path.expanduser("~/auto_trader/V3.6/")
STATE_FILE = os.path.join(PROJECT_DIR, "autotrader_state.json")
TRADES_CSV = os.path.join(PROJECT_DIR, "autotrader_trades.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "autotrader.log")
ENV_FILE = os.path.join(PROJECT_DIR, ".env")

# default config (overridden by .env)
CFG = {
    "API_KEY": "",
    "API_SECRET": "",
    "SYMBOLS": "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT",
    "QUOTE": "USDT",
    "POSITION_USD": "25",         # USD per buy
    "FAST_SMA": "20",
    "SLOW_SMA": "50",
    "TIMEFRAME": "15m",
    "POLL_SECONDS": "30",
    "TP_PCT": "1.0",
    "SL_PCT": "0.8",
    "DRY_RUN": "true",            # safe default
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
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
    # read from .env config
    symbols = [s.strip().upper().replace("/", "") for s in CFG["SYMBOLS"].split(",") if s.strip()]
    if not symbols:
        raise ValueError("No SYMBOLS defined in CFG['SYMBOLS']")

    # Binance expects JSON array string in query

    symbols_param = urllib.parse.quote(json.dumps(symbols, separators=(',', ':')))
    info = http_get(f"/api/v3/exchangeInfo?symbols={symbols_param}")


    filters = {}
    for s in info["symbols"]:
        lot = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
        notional = next(f for f in s["filters"] if f["filterType"] == "NOTIONAL")
        filters[s["symbol"]] = {
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
            subprocess.run(["termux-notification", "--title", title, "--content", message[:4096]], check=False)
        except Exception as e:
            logging.warning("termux-notification failed: %s", e)
    else:
        # fallback log
        logging.info("[NOTIFY] %s: %s", title, message)

# ---------- Binance HTTP helpers ----------
# AFTER (The simple, correct fix)
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

# create market order:
# - for BUY we use quoteOrderQty to specify how much QUOTE (USDT) we want to spend
# - for SELL we specify quantity (base asset)
def create_market_order(api_key, api_secret, symbol, side, quantity=None, quoteOrderQty=None):
    """Create a MARKET order with safe decimal formatting (no scientific notation)."""

    params = {"symbol": symbol.replace("/", ""), "side": side, "type": "MARKET"}

    # Use quoteOrderQty if provided (e.g., buying with USDT amount)
    if quoteOrderQty is not None:
        params["quoteOrderQty"] = "{:.8f}".format(float(quoteOrderQty)).rstrip("0").rstrip(".")

    # Otherwise use quantity, formatted to the correct precision
    elif quantity is not None:
        f = SYMBOL_FILTERS[symbol.replace("/", "")]
        decimals = int(round(-math.log10(f["stepSize"])))   # derive allowed decimals from stepSize
        params["quantity"] = f"{float(quantity):.{decimals}f}".rstrip("0").rstrip(".")

    else:
        raise ValueError("quantity or quoteOrderQty required")

    return http_post("/api/v3/order", params=params, signed=True, api_key=api_key, api_secret=api_secret)
def safe_market_buy(api_key, api_secret, symbol, position_usd):
    """
    Try to place a MARKET BUY using quoteOrderQty.
    If Binance rejects it, fallback to manual qty calculation.
    """
    last_price = fetch_price(symbol)

    # 1. Try quoteOrderQty
    try:
        return create_market_order(api_key, api_secret, symbol, "BUY", quoteOrderQty=position_usd)
    except Exception as e:
        logging.warning("%s: quoteOrderQty=%s failed (%s), falling back to quantity", symbol, position_usd, e)

    # 2. Fallback: calculate quantity manually
    raw_qty = position_usd / last_price
    qty = math.floor(raw_qty * 1e6) / 1e6

    try:
        return create_market_order(api_key, api_secret, symbol, "BUY", quantity=qty)
    except Exception as e:
        logging.error("%s: both quoteOrderQty and quantity BUY failed (%s)", symbol, e)
        return None
def safe_market_sell(api_key, api_secret, symbol, base_quantity):
    """
    Sell base_quantity respecting Binance stepSize, minQty, minNotional,
    and return a dict compatible with run_cycle.
    """
    f = SYMBOL_FILTERS[symbol.replace("/", "")]
    # Round DOWN to nearest stepSize
    qty = math.floor(base_quantity / f["stepSize"]) * f["stepSize"]

    # Determine allowed decimal places from stepSize
    decimals = int(round(-math.log10(f["stepSize"])))
    qty = float(f"%.{decimals}f" % qty)

    if qty < f["minQty"]:
        logging.warning("%s: quantity %.8f below minQty %.8f, skipping sell", symbol, qty, f["minQty"])
        return None

    # Calculate notional
    price = fetch_price(symbol)
    notional = qty * price
    if notional < f["minNotional"]:
        logging.warning("%s: notional %.2f below minNotional %.2f, skipping sell", symbol, notional, f["minNotional"])
        return None

    # Place the sell order
    order_info = create_market_order(api_key, api_secret, symbol, "SELL", quantity=qty)

    # Ensure executedQty and cummulativeQuoteQty exist in dict
    executed_qty = float(order_info.get("executedQty", qty))
    executed_quote = float(order_info.get("cummulativeQuoteQty", qty * price))

    return {
        "executedQty": executed_qty,
        "cummulativeQuoteQty": executed_quote,
        "status": order_info.get("status", "UNKNOWN"),
    }
# ---------- SMA & signal (pure python) ----------
def simple_sma(prices, n):
    if len(prices) < n:
        return None
    return sum(prices[-n:]) / n

def generate_signal_with_atr(klines, atr_period, atr_mult
, ema_period):
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]

    if len(closes) < max(atr_period, ema_period) + 1:
        return "HOLD"

    # Calculate ATR
    atr = calculate_atr(klines[:-1], period=int(CFG["ATR_PERIOD"]))

    # Trend filter (EMA)
    alpha = 2 / (ema_period + 1)
    ema = closes[0]
    for price in closes[1:]:
        ema = alpha * price + (1 - alpha) * ema

    recent_high = max(highs[-atr_period-1:-1])
    recent_low = min(lows[-atr_period-1:-1])
    last_close = closes[-1]

    #SMA PERCENT DIFF CALCULATION
    min_diff_pct = float(CFG.get("MIN_DIFF_PCT", 0.1))
    f_curr = simple_sma(closes, int(CFG["FAST_SMA"]))
    s_curr = simple_sma(closes, int(CFG["SLOW_SMA"]))

    if f_curr is None or s_curr is None or s_curr == 0:
        return "HOLD"

    diff_pct = ((f_curr - s_curr) / s_curr) * 100.0

     # Debug logging
    logger.info(
        f"[ATR-SIGNAL] close={last_close:.2f}, ema={ema:.2f}, "
        f"recent_high={recent_high:.2f}, recent_low={recent_low:.2f}, "
        f"atr={atr:.2f}, mult={atr_mult}, "
        f"buy_trigger={recent_high + atr_mult * atr:.2f}, "
        f"sell_trigger={recent_low - atr_mult * atr:.2f}"
    )

    # Entry rules
    if (last_close > recent_high + atr_mult * atr and last_close > ema) or (diff_pct >= min_diff_pct and last_close > ema):
        return "BUY"
    elif last_close < recent_low - atr_mult * atr and last_close < ema:
        return "SELL"
    else:
        return "HOLD"

def calculate_atr(klines, period=14):
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]

    trs = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_close, low_close)
        trs.append(tr)

    return sum(trs[-period:]) / period if len(trs) >= period else None

def update_dynamic_levels(state, current_price, tp_step=0.6, sl_step=0.8):
    """
    Adjusts take-profit and stop-loss dynamically when price moves up.

    Args:
        state (dict): current state for this symbol (buy_price, tp, sl).
        current_price (float): latest market price.
        tp_step (float): % distance for take profit (default 0.8%).
        sl_step (float): % distance for stop loss (default 0.8%).

    Returns:
        dict: updated state with new tp and sl.
    """
    if "buy_price" not in state or state["buy_price"] == 0:
        return state  # nothing to update if not in trade

    # If price moves higher than last take-profit, raise both TP and SL
    if current_price > state["tp"]:
        state["tp"] = current_price * (1 + tp_step / 100)
        state["sl"] = current_price * (1 - (sl_step/2) / 100)

    return state
# ---------- State & logging ----------
def read_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            logging.exception("Failed to read state")
    return {"positions": {}}

def write_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)

def append_trade(row: dict):
    newfile = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if newfile:
            writer.writerow(list(row.keys()))
        writer.writerow([row[k] for k in row.keys()])

def summary_24h():
    if not os.path.exists(TRADES_CSV):
        notify("AutoTrader 24h Summary", "No trades recorded yet.")
        return
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    count = 0
    pnl = 0.0
    vol = 0.0
    with open(TRADES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # expected time stored in ISO format (UTC)
            try:
                t = datetime.fromisoformat(r["time"])
            except Exception:
                continue
            if t.replace(tzinfo=timezone.utc) >= cutoff:
                count += 1
                pnl += float(r.get("pnl_usd") or 0.0)
                vol += float(r.get("notional_usd") or 0.0)
    notify("AutoTrader 24h", f"Trades: {count} | Notional: ${vol:.2f} | PnL: ${pnl:.2f}")

# ---------- Engine ----------
def run_cycle(cfg, state):
    api_key = cfg["API_KEY"]
    api_secret = cfg["API_SECRET"]
    symbols = [s.strip().upper() for s in cfg["SYMBOLS"].split(",") if s.strip()]
    tf = cfg["TIMEFRAME"]
    position_usd = float(cfg["POSITION_USD"])
    dry_run = str(cfg["DRY_RUN"]).lower() in ("1", "true", "yes", "y")
    poll = int(cfg["POLL_SECONDS"])

    fast = int(cfg["FAST_SMA"])
    slow = int(cfg["SLOW_SMA"])

    for symbol in symbols:
        try:
            # fetch klines and closes
            klines = fetch_klines(symbol, cfg["TIMEFRAME"], limit=60)
            atr = calculate_atr(klines)
            closes = [float(k[4]) for k in klines]
            sig = generate_signal_with_atr(klines, atr_period=int(CFG["ATR_PERIOD"]), atr_mult=float(CFG["ATR_MULT"]), ema_period=int(CFG["EMA_PERIOD"]))
            last_price = closes[-1]
            pos = state["positions"].get(symbol)
            logging.info("%s: price=%.6f signal=%s pos=%s", symbol, last_price, sig, "OPEN" if pos else "NONE")

            min_diff_pct = float(cfg.get("MIN_DIFF_PCT", 0.1))


            if sig == "BUY" and pos is None:
                # Market buy using quoteOrderQty to spend position_usd USDT
                if dry_run:
                    # simulate execution: compute amount = position_usd / last_price
                    amount = position_usd / last_price
                    executed_qty = amount
                    executed_quote = position_usd
                    avg_price = last_price
                    order_info = {"status": "SIMULATED", "executedQty": str(executed_qty), "cummulativeQuoteQty": str(executed_quote)}
                else:
                    order_info = safe_market_buy(api_key, api_secret, symbol, position_usd)
                    if not order_info:
                        logging.error("%s: BUY order failed completely, skipping", symbol)
                        continue
                    # order_info includes executedQty and cummulativeQuoteQty
                    executed_qty = float(order_info.get("executedQty", "0"))
                    executed_quote = float(order_info.get("cummulativeQuoteQty", "0") or order_info.get("cummulativeQuoteQty", executed_qty * last_price))
                    avg_price = (executed_quote / executed_qty) if executed_qty else last_price

                amount_bought = float(order_info.get("executedQty", executed_qty))
                notional = float(order_info.get("cummulativeQuoteQty", executed_quote))
                entry_price = avg_price

                # store position
                atr_period = int(cfg.get("ATR_PERIOD", 14))
                atr_mult = float(cfg.get("ATR_MULTIPLIER", 2.0))
                tp_mult = float(cfg.get("TP_MULTIPLIER", 1.5))

                atr_val = calculate_atr(klines, atr_period)
                if atr_val is None:
                    logging.warning("%s: ATR not available, skipping trade", symbol)
                    return

                state["positions"][symbol] = {
                    "amount": amount_bought,
                    "entry_price": entry_price,
                    "tp": entry_price + (atr_val * tp_mult),
                    "sl": entry_price - (atr_val * atr_mult),
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                }
                write_state(state)
                msg = f"{symbol} BUY executed {amount_bought:.8f} @ {entry_price:.6f} (notional ${notional:.2f})"
                logging.info(msg)
                notify("Trade Placed", msg)
                append_trade({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "side": "BUY",
                    "price": f"{entry_price:.6f}",
                    "amount": f"{amount_bought:.8f}",
                    "notional_usd": f"{notional:.2f}",
                    "pnl_usd": "0.0",
                    "dry_run": str(dry_run),
                })

            elif pos is not None:
                # manage existing position: exit on TP/SL or SELL signal
                amount = float(pos["amount"])
                entry = float(pos["entry_price"])

                # ATR-based trailing SL
                atr_period = int(cfg.get("ATR_PERIOD", 14))
                atr_mult = float(cfg.get("ATR_MULTIPLIER", 2.0))
                atr_val = calculate_atr(klines, atr_period)

                if atr_val is not None and last_price > entry:
                    new_sl = last_price - (atr_val * atr_mult)
                