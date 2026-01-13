#!/usr/bin/env python3
"""
V7
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
from decimal import Decimal, ROUND_DOWN, getcontext
import threading
import copy
import tempfile
try:
    import numpy as np
except Exception:
    def _safe_mean(arr):
        try:
            a = [float(x) for x in arr]
            return sum(a) / len(a) if a else 0.0
        except Exception:
            return 0.0
    class _DummyNP:
        mean = staticmethod(_safe_mean)
    np = _DummyNP()

# ML Strategy imports (for cycle4)
try:
    import pandas as pd
    import numpy as np
    import joblib
    from feature_engineering import engineer_features
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning("ML dependencies not available: %s", e)
    ML_AVAILABLE = False
    pd = None
    np = None
    joblib = None

# ONNX runtime for mobile deployment (optional)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logging.info("✓ ONNX runtime available - mobile ML enabled")
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

# Pure Python inference for ARM/mobile devices (no deps needed)
try:
    from lightgbm_inference import LightGBMInference
    JSON_INFERENCE_AVAILABLE = True
    logging.info("✓ Pure Python inference available - ARM/mobile compatible")
except ImportError:
    JSON_INFERENCE_AVAILABLE = False

_write_lock = threading.Lock()
# increased decimal precision
getcontext().prec = 28
# Try termux wake lock (only call if binary exists — safer on non-Termux systems)
try:
    if shutil.which("termux-wake-lock"):
        subprocess.run(["termux-wake-lock"], check=False)
except Exception:
    pass
# ---------- CONFIG / FILES ----------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(PROJECT_DIR, "autotrader_state.json")
TRADES_CSV = os.path.join(PROJECT_DIR, "autotrader_trades.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "autotrader.log")
ENV_FILE = os.path.join(PROJECT_DIR, ".env")
# OPTIMAL CONFIGURATION - V8 Final (Volume Optimized)
# Performance: $13.73 (+1.37%) | 18 trades | 77.8% WR | PF 4.69
CFG = {
    "API_KEY": "",
    "API_SECRET": "",
    "SYMBOLS": "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT,ADA/USDT,DOGE/USDT,AVAX/USDT,DOT/USDT,MATIC/USDT,LTC/USDT,LINK/USDT",
    "QUOTE": "USDT",
    "POSITION_USD": "20",
    "FAST_SMA": "20",
    "SLOW_SMA": "50",
    "TIMEFRAME": "15m",
    "POLL_SECONDS": "30",
    "TP_PCT": "1.0",
    "SL_PCT": "0.8",
    "DRY_RUN": "true",
    "NOTIFY": "true",
    "EMA_FAST": "7",
    "EMA_SLOW": "21",
    "ATR_PERIOD": "10",
    "ATR_MULT": "2.0",
    "ATR_BUFFER_MULT": "0.3",
    "ATR_ADAPTIVE_ENABLED": "true",
    "ATR_ADAPTIVE_REDUCTION": "0.2",
    "TAKE_RATIO": "4.0",
    "VOLUME_MULT": "1.8",
    "MIN_DIFF_PCT": "0.05",
    "EMA_TREND_PERCENT": "0.05",
    "COOLDOWN_SECONDS": "300",
    "FEE_PCT": "0.001",
    "SLIPPAGE_PCT": "0.001",
    "POSITION_PCT": "0.0",
    "MIN_POSITION_USD": "5",
    "CHANDELIER_MULT": "1.5",
    "BUY_WINDOW_CANDLES": "10",
    "RSI_PERIOD": "10",
    "RSI_THRESHOLD": "25",
    "BB_PERIOD": "20",
    "PERCENT_TRAIL": "4.5",
}
BINANCE_BASE = "https://api.binance.com"
# ---------- Logging ----------
os.makedirs(PROJECT_DIR, exist_ok=True)
from logging.handlers import RotatingFileHandler

def _setup_logging():
    """Initialize or reinitialize logging after PROJECT_DIR is determined."""
    global file_handler, stream_handler, root_logger, logger
    
    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    # Small, efficient log format to reduce size (level shortened to 1 char)
    log_formatter = logging.Formatter("%(asctime)s | %(levelname).1s | %(message)s")
    
    # Rotating file handler (keeps on-disk logs bounded)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=int(CFG.get("LOG_MAX_BYTES", 1000000)),
        backupCount=int(CFG.get("LOG_BACKUP_COUNT", 3)),
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler set to WARNING to avoid verbose stdout spam on Android
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.WARNING)      # Set to WARNING for reduced log output to terminal
    
    # Apply handlers to the root logger so all logging calls are captured
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    logger = logging.getLogger(__name__)

# Initial logging setup
_setup_logging()
# Simple rate-limited logger for very noisy messages (keyed by caller)
_last_log = {}
def log_rate_limited(key, level, msg, *args, interval=30, **kwargs):
    now = time.time()
    last = _last_log.get(key, 0)
    if now - last >= interval:
        _last_log[key] = now
        root_logger.log(level, msg, *args, **kwargs)
def check_user_command(cfg, state):
    """
    Read single-line command from PROJECT_DIR/command.txt and execute it once.
    Accepts:
      BUY SYMBOL [USD_amount]
      SELL SYMBOL
    Behavior:
      - Clears command.txt immediately (prevents double execution).
      - Writes a small audit file command.txt.last with the original command and result.
      - Ensures SYMBOL_FILTERS is available before attempting live orders.
      - Does not remove state if a SELL fails.
    """
    try:
        cmd_path = os.path.join(PROJECT_DIR, "command.txt")
        audit_path = cmd_path + ".last"

        # ensure file exists
        if not os.path.exists(cmd_path):
            with open(cmd_path, "w") as f:
                f.write("")
            return

        # read and trim
        with open(cmd_path, "r") as f:
            line = f.read().strip()

        if not line:
            return

        logging.info("[USER COMMAND] %s", line)

        # clear command file immediately to avoid duplicate processing
        try:
            with open(cmd_path, "w") as f:
                f.write("")
        except Exception:
            logging.exception("Failed to clear command file; continuing (may re-run)")

        # ensure SYMBOL_FILTERS is loaded (safe no-op if already present)
        global SYMBOL_FILTERS
        if not SYMBOL_FILTERS:
            try:
                SYMBOL_FILTERS = load_symbol_filters(api_key=cfg.get("API_KEY"), api_secret=cfg.get("API_SECRET"))
            except Exception:
                logging.exception("Failed to load SYMBOL_FILTERS for user command")

        parts = line.split()
        if len(parts) < 2:
            logging.error("Invalid user command format: '%s'", line)
            with open(audit_path, "a") as a:
                a.write(f"{datetime.now(timezone.utc).isoformat()}  INVALID_FORMAT  {line}\n")
            return

        action = parts[0].upper()
        symbol_raw = parts[1].upper()
        
        # Normalize symbol format: strip "/" and add it back before quote currency
        # This prevents duplicate positions from "BTC/USDT" vs "BTCUSDT"
        symbol_no_slash = symbol_raw.replace("/", "")
        quote = cfg.get("QUOTE", "USDT")
        
        # Add slash before quote currency to match automated system format
        if symbol_no_slash.endswith(quote):
            base = symbol_no_slash[:-len(quote)]
            symbol = f"{base}/{quote}"
        else:
            # If quote not found, try common alternatives
            for alt_quote in ["USDT", "USDC", "BUSD", "BTC", "ETH"]:
                if symbol_no_slash.endswith(alt_quote):
                    base = symbol_no_slash[:-len(alt_quote)]
                    symbol = f"{base}/{alt_quote}"
                    break
            else:
                # Fallback: use as-is if no quote currency detected
                symbol = symbol_raw
                logging.warning("Could not normalize symbol '%s', using as-is", symbol_raw)

        if action not in ("BUY", "SELL"):
            logging.error("Invalid user command action: '%s'", action)
            with open(audit_path, "a") as a:
                a.write(f"{datetime.now(timezone.utc).isoformat()}  INVALID_ACTION  {line}\n")
            return

        api_key = cfg.get("API_KEY")
        api_secret = cfg.get("API_SECRET")
        dry_run = str(cfg.get("DRY_RUN", "true")).lower() in ("1", "true", "yes", "y")

        # ----------------- BUY -----------------
        # ----------------- BUY -----------------
        if action == "BUY":
            # parse USD amount or fall back to POSITION_USD
            if len(parts) >= 3:
                try:
                    amount_usd = float(parts[2])
                except Exception:
                    logging.error("Invalid amount in BUY command: %s", parts[2])
                    with open(audit_path, "a") as a:
                        a.write(f"{datetime.now(timezone.utc).isoformat()}  INVALID_AMOUNT  {line}\n")
                    return
            else:
                amount_usd = float(cfg.get("POSITION_USD", 25))

            price = fetch_price(symbol)
            if price is None:
                logging.error("Could not fetch price for %s BUY command", symbol)
                with open(audit_path, "a") as a:
                    a.write(f"{datetime.now(timezone.utc).isoformat()}  PRICE_FETCH_FAIL  {line}\n")
                return

            if dry_run:
                executed_qty = amount_usd / price if price else 0.0
                executed_quote = amount_usd
                order_status = "SIMULATED"
            else:
                order = create_market_order(api_key, api_secret, symbol, "BUY", quoteOrderQty=amount_usd)
                if not order:
                    logging.error("User BUY order failed for %s", symbol)
                    with open(audit_path, "a") as a:
                        a.write(f"{datetime.now(timezone.utc).isoformat()}  BUY_ORDER_FAIL  {line}\n")
                    return
                executed_qty = float(order.get("executedQty", 0) or 0)
                executed_quote = float(order.get("cummulativeQuoteQty", 0) or 0)
                order_status = order.get("status", "UNKNOWN")

            if executed_qty <= 0:
                logging.error("BUY executed zero quantity for %s; aborting state update", symbol)
                with open(audit_path, "a") as a:
                    a.write(f"{datetime.now(timezone.utc).isoformat()}  BUY_ZERO_QTY  {line}\n")
                return

            avg_price = (executed_quote / executed_qty) if executed_qty and executed_quote else price

            # --- compute ATR properly for manual position (use same period and klines as auto)
            tf = cfg.get("TIMEFRAME", "15m")
            atr_period = int(get_cfg("ATR_PERIOD", int, 14))
            klines_for_atr = fetch_klines(symbol, tf, limit=200) or []
            atr_val = None
            try:
                atr_val = calculate_atr(klines_for_atr, atr_period)
            except Exception:
                atr_val = None
            if not atr_val:
                # fallback tiny floor (matches update_dynamic_sl_pro fallback)
                min_atr_pct = 0.0002
                atr_val = max(avg_price * min_atr_pct, 1e-8)

            # consistent SL/TP computation using same configuration as automated buys
            atr_mult_cfg = float(get_cfg("ATR_MULT", float, 2.0))
            tp_mult = float(get_cfg("TAKE_RATIO", float, 1.5))
            fee_pct = float(get_cfg("FEE_PCT", float, 0.001))
            slip_pct = float(get_cfg("SLIPPAGE_PCT", float, 0.001))
            buffer_mult = 1.0 + fee_pct + slip_pct
            # Expand SL to cover expected fees/slippage, and reduce TP to account for costs
            initial_sl = avg_price - (atr_val * atr_mult_cfg) * buffer_mult
            initial_sl = max(initial_sl, avg_price * 0.98)  # preserve the 2% safety floor used in SL function
            initial_tp = avg_price + ((atr_val * atr_mult_cfg) * tp_mult) * max(0.0, (1.0 - fee_pct - slip_pct))

            # safe insert into state (fields match auto-created positions)
            state.setdefault("positions", {})[symbol] = {
                "amount": executed_qty,
                "entry_price": avg_price,
                "sl": initial_sl,
                "initial_sl": initial_sl,
                "tp": initial_tp,
                "initial_tp": initial_tp,
                "locked_atr": float(atr_val),
                "risk": float(atr_val * atr_mult_cfg),
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "buy_signal": "MANUAL",
                "highest_since_entry": avg_price,
                "last_ema_slope": 0.0,
            }

            write_state(state)

            # Record the manual buy in trades CSV
            append_trade({
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": "BUY",
                "reason": "MANUAL",
                "price": f"{avg_price:.6f}",
                "amount": f"{executed_qty:.8f}",
                "notional_usd": f"{executed_quote:.2f}",
                "pnl_usd": "0.0",
                "dry_run": str(dry_run),
            })

            msg = f"USER BUY {symbol} {executed_qty:.8f} @ {avg_price:.6f} (status={order_status}) locked_atr={atr_val:.6f}"
            logging.info(msg)
            notify("User BUY", msg)
            with open(audit_path, "a") as a:
                a.write(f"{datetime.now(timezone.utc).isoformat()}  BUY_OK  {line}  -> {msg}\n")
            return

        # ----------------- SELL -----------------
        if action == "SELL":
            pos = state.get("positions", {}).get(symbol)
            if not pos:
                logging.warning("No open position to SELL for %s", symbol)
                with open(audit_path, "a") as a:
                    a.write(f"{datetime.now(timezone.utc).isoformat()}  SELL_NO_POSITION  {line}\n")
                return

            qty = float(pos.get("amount", 0.0))
            if qty <= 0:
                logging.warning("Position amount for %s is zero, nothing to sell", symbol)
                with open(audit_path, "a") as a:
                    a.write(f"{datetime.now(timezone.utc).isoformat()}  SELL_ZERO_AMOUNT  {line}\n")
                return

            price = fetch_price(symbol) or 0.0

            if dry_run:
                executed_qty = qty
                executed_quote = executed_qty * price
                order_status = "SIMULATED"
            else:
                exit_order = safe_market_sell(api_key, api_secret, symbol, qty, state)
                if not exit_order:
                    logging.error("User SELL order failed for %s; REMOVING position to prevent zombie", symbol)
                    with open(audit_path, "a") as a:
                        a.write(f"{datetime.now(timezone.utc).isoformat()}  SELL_ORDER_FAIL  {line}\n")
                    # Critical: remove position even if order failed, to prevent zombie accumulation
                    state.get("positions", {}).pop(symbol, None)
                    state.setdefault("cooldowns", {})[symbol] = datetime.now(timezone.utc).isoformat()
                    safe_write_state(state)
                    return
                executed_qty = float(exit_order.get("executedQty", 0) or 0)
                executed_quote = float(exit_order.get("cummulativeQuoteQty", executed_qty * price))
                order_status = exit_order.get("status", "UNKNOWN")

            if executed_qty <= 0:
                logging.error("SELL executed zero quantity for %s; REMOVING position to prevent zombie", symbol)
                with open(audit_path, "a") as a:
                    a.write(f"{datetime.now(timezone.utc).isoformat()}  SELL_ZERO_QTY  {line}\n")
                # Remove position to prevent zombie accumulation
                state.get("positions", {}).pop(symbol, None)
                state.setdefault("cooldowns", {})[symbol] = datetime.now(timezone.utc).isoformat()
                safe_write_state(state)
                return

            # compute PnL using stored entry_price where possible
            entry_price = float(pos.get("entry_price", 0.0) or 0.0)
            pnl = ((executed_quote / executed_qty) - entry_price) * executed_qty if executed_qty and entry_price else (executed_quote - (entry_price * executed_qty))

            # record trade and remove position only after success
            append_trade({
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": "SELL",
                "reason": "MANUAL",
                "price": f"{(executed_quote / executed_qty) if executed_qty else price:.6f}",
                "amount": f"{executed_qty:.8f}",
                "notional_usd": f"{executed_quote:.2f}",
                "pnl_usd": f"{pnl:.2f}",
                "dry_run": str(dry_run),
            })

            msg = f"USER SELL {symbol} {executed_qty:.8f} @ {(executed_quote / executed_qty) if executed_qty else price:.6f}  PnL ${pnl:.2f} (status={order_status})"
            logging.info(msg)
            notify("User SELL", msg)

            # remove position from state
            try:
                state.get("positions", {}).pop(symbol, None)
                state.setdefault("cooldowns", {})[symbol] = datetime.now(timezone.utc).isoformat()
                write_state(state)
            except Exception:
                logging.exception("Failed to remove position from state after SELL for %s", symbol)

            with open(audit_path, "a") as a:
                a.write(f"{datetime.now(timezone.utc).isoformat()}  SELL_OK  {line}  -> {msg}\n")
            return

    except Exception:
        logging.exception("Unexpected error in check_user_command")
        try:
            with open(os.path.join(PROJECT_DIR, "command.txt.error"), "a") as efile:
                efile.write(f"{datetime.now(timezone.utc).isoformat()}  EXC  {line if 'line' in locals() else ''}\n")
        except Exception:
            pass
# ---------- Utilities ----------
def load_env_file(path):
    if not os.path.exists(path):
        logging.debug("No .env found at %s — using defaults (DRY_RUN=%s)", path, CFG.get("DRY_RUN"))
        return
    seen = set()
    overrides = []
    duplicates = []
    with open(path, "r") as f:
        for line in f:
            if "=" not in line or line.strip().startswith("#"):
                continue
            k, v = line.strip().split("=", 1)
            k = k.strip()
            v = v.strip().strip('\"').strip("'")
            # Track overrides and duplicates without noisy per-line warnings
            if k in CFG and k not in seen:
                overrides.append(k)
            elif k in seen:
                duplicates.append(k)
            seen.add(k)
            CFG[k] = v
            os.environ[k] = v

    # Intentionally do not log per-key overrides or duplicates —
    # .env overrides are expected behaviour and should be silent.
    # (Keep parsing and applying values only.)
def get_cfg(key, cast=str, default=None):
    v = CFG.get(key, default)
    if v is None:
        return default
    if cast is bool:
        return str(v).lower() in ("1", "true", "yes", "y")
    try:
        return cast(v)
    except Exception:
        return v
# ---------- HTTP helpers ----------
def http_get(path, params=None, signed=False, api_key=None, api_secret=None,
             max_retries=5, base_delay=5, timeout=30):
    if params is None:
        params = {}

    if signed:
        if not api_secret:
            raise RuntimeError("API secret required for signed request")
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        signature = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        query = f"{query}&signature={signature}"
    else:
        query = urllib.parse.urlencode(params)

    base_url = BINANCE_BASE + path
    sep = "&" if "?" in base_url else "?"
    url = f"{base_url}{sep}{query}" if query else base_url

    headers = {"User-Agent": "AutoTrader/1.0"}
    if signed and api_key:
        headers["X-MBX-APIKEY"] = api_key

    req = urllib.request.Request(url, headers=headers, method="GET")
    attempt = 0
    while attempt < max_retries:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode()
                return json.loads(data)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()
            except Exception:
                pass
            attempt += 1
            log_rate_limited(f"http_get_err_{getattr(e, 'code', 'unknown')}", logging.WARNING,
                             "HTTP GET %s failed (attempt %d): %s %s", url, attempt, getattr(e, "code", ""), body,
                             interval=30)
            code = getattr(e, "code", None)
            # don't retry on 4xx (except 429)
            if code and 400 <= code < 500 and code not in (429,):
                logging.error("Client error %s for %s: not retrying", code, url)
                return None
            # special-case known Binance codes in body
            try:
                j = json.loads(body)
                if j.get("code") in (-1003, -1015, -1021):
                    time.sleep(10)
            except Exception:
                pass
        except urllib.error.URLError as e:
            attempt += 1
            log_rate_limited("http_get_urlerr", logging.WARNING,
                             "HTTP GET %s failed (attempt %d): %s", url, attempt, e,
                             interval=30)
        except Exception as e:
            attempt += 1
            # Log full exception but rate-limit to avoid huge repeated traces
            log_rate_limited("http_get_exc", logging.ERROR,
                             "Unexpected error on %s (attempt %d): %s", url, attempt, e,
                             interval=60)

        if attempt >= max_retries:
            logging.error("Failed to fetch %s after %d attempts", url, attempt)
            return None

        delay = min(base_delay * (2 ** (attempt - 1)), 60)
        time.sleep(delay)
def http_post(path, params=None, signed=False, api_key=None, api_secret=None, timeout=30):
    if params is None:
        params = {}
    params_to_sign = dict(params)
    if signed:
        if not api_secret:
            raise RuntimeError("API secret required for signed request")
        params_to_sign["timestamp"] = int(time.time() * 1000)
    query = urllib.parse.urlencode(params_to_sign)
    if signed:
        signature = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        query = query + "&signature=" + signature
    base_url = BINANCE_BASE + path
    sep = "&" if "?" in base_url else "?"
    url = f"{base_url}{sep}{query}" if query else base_url
    headers = {"User-Agent": "AutoTrader/1.0"}
    if api_key:
        headers["X-MBX-APIKEY"] = api_key
    req = urllib.request.Request(url, headers=headers, method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode()
            return json.loads(data)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else ""
        log_rate_limited("http_post_err", logging.WARNING,
                         "HTTP POST %s failed: %s %s", url, getattr(e, "code", ""), body,
                         interval=30)
        return None
    except Exception as e:
        log_rate_limited("http_post_exc", logging.ERROR, "HTTP POST error: %s", e, interval=60)
        return None
# ---------- Market helpers ----------
def fetch_klines(symbol, interval, limit=200):
    params = {"symbol": symbol.replace("/", ""), "interval": interval, "limit": limit}
    return http_get("/api/v3/klines", params=params)
def fetch_price(symbol):
    params = {"symbol": symbol.replace("/", "")}
    t = http_get("/api/v3/ticker/price", params=params)
    if not t:
        logging.warning("fetch_price returned None for %s", symbol)
        return None
    try:
        price = float(t.get("price", 0))
        if price <= 0:
            logging.error("fetch_price returned invalid price %.8f for %s", price, symbol)
            return None
        return price
    except (ValueError, TypeError) as e:
        logging.error("Failed to parse price for %s: %s", symbol, e)
        return None
def fetch_balance(api_key, api_secret, asset):
    res = http_get("/api/v3/account", signed=True, api_key=api_key, api_secret=api_secret)
    if not res:
        logging.warning("fetch_balance returned None for asset %s", asset)
        return None
    for b in res.get("balances", []):
        if b.get("asset") == asset:
            try:
                bal = float(b.get("free", 0.0))
                return bal
            except Exception as e:
                logging.error("Failed to parse balance for %s: %s", asset, e)
                return None
    logging.warning("Asset %s not found in account balances", asset)
    return None
def load_symbol_filters(api_key=None, api_secret=None):
    symbols = [s.strip().upper().replace("/", "") for s in CFG["SYMBOLS"].split(",") if s.strip()]
    if not symbols:
        raise ValueError("No SYMBOLS defined in CFG['SYMBOLS']")
    try:
        res = http_get("/api/v3/exchangeInfo", params={"symbols": json.dumps(symbols, separators=(',', ':'))})
        if not res:
            logging.error("Failed to load exchangeInfo")
            return {}
        filters = {}
        for s in res.get("symbols", []):
            lot = next((f for f in s.get("filters", []) if f.get("filterType") == "LOT_SIZE"), None)
            notional = next((f for f in s.get("filters", []) if f.get("filterType") == "NOTIONAL"), None)
            if not lot or not notional:
                continue
            try:
                filters[s["symbol"]] = {
                    "minQty": float(lot["minQty"]),
                    "stepSize": float(lot["stepSize"]),
                    "minNotional": float(notional["minNotional"]),
                }
            except Exception:
                continue
        return filters
    except Exception as e:
        logging.exception("load_symbol_filters failed: %s", e)
        return {}
SYMBOL_FILTERS = {}
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
        logging.info("[NOTIFY] %s: %s", title, message)
def compute_position_usd(api_key, api_secret):
    """Compute position USD amount using percent-of-equity if configured.
    Returns a float USD amount or None if below minimum configured value.
    """
    try:
        pct = float(get_cfg("POSITION_PCT", float, 0.0))
    except Exception:
        pct = 0.0
    try:
        min_pos = float(get_cfg("MIN_POSITION_USD", float, 5.0))
    except Exception:
        min_pos = 5.0

    if pct and pct > 0:
        # Use quote asset balance as equity estimate
        quote = CFG.get("QUOTE", "USDT")
        try:
            bal = fetch_balance(api_key, api_secret, quote)
        except Exception:
            bal = None
        if bal is None:
            logging.warning("Could not fetch %s balance for position sizing; falling back to POSITION_USD", quote)
            pos = float(get_cfg("POSITION_USD", float, 25))
        else:
            pos = bal * pct
    else:
        pos = float(get_cfg("POSITION_USD", float, 25))

    if pos < min_pos:
        logging.warning("Calculated position USD $%.2f below MIN_POSITION_USD $%.2f", pos, min_pos)
        return None
    return pos
# --- create order with Decimal quantization ---
def create_market_order(api_key, api_secret, symbol, side, quantity=None, quoteOrderQty=None):
    params = {"symbol": symbol.replace("/", ""), "side": side, "type": "MARKET"}
    if quoteOrderQty is not None:
        params["quoteOrderQty"] = "{:.8f}".format(float(quoteOrderQty)).rstrip("0").rstrip(".")
    elif quantity is not None:
        f = SYMBOL_FILTERS.get(symbol.replace("/", ""))
        if not f:
            logging.error("create_market_order: Symbol filters not available for %s; cannot quantize quantity", symbol)
            return None
        step = f["stepSize"]
        step_s = f"{step:.8f}".rstrip("0").rstrip(".")
        if "." in step_s:
            decimals = len(step_s.split(".")[1])
        else:
            decimals = 0
        qty_dec = Decimal(str(float(quantity))).quantize(Decimal(10) ** -decimals, rounding=ROUND_DOWN)
        params["quantity"] = format(qty_dec.normalize(), 'f')
    else:
        raise ValueError("quantity or quoteOrderQty required")
    return http_post("/api/v3/order", params=params, signed=True, api_key=api_key, api_secret=api_secret)
def safe_market_buy(api_key, api_secret, symbol, position_usd):
    try:
        return create_market_order(api_key, api_secret, symbol, "BUY", quoteOrderQty=position_usd)
    except Exception as e:
        logging.error("%s: MARKET BUY failed (%s)", symbol, e)
        return None
def safe_market_sell(api_key, api_secret, symbol, base_quantity, state=None):
    
    asset = symbol.replace("/", "").replace("USDT", "")
    f = SYMBOL_FILTERS.get(symbol.replace("/", ""))
    if not f:
        logging.error("No symbol filters for %s", symbol)
        return None
    free_bal = fetch_balance(api_key, api_secret, asset)
    if free_bal is None:
        logging.error("%s: Balance API returned None for %s; aborting sell to prevent state corruption.", symbol, asset)
        return None
    base_qty_float = float(base_quantity)
    if free_bal < 0.0001:
        logging.error(
            "%s: Balance API returned %.8f (expected %.8f). "
            "Aborting sell to prevent state corruption. Will retry next cycle.",
            symbol, free_bal, base_qty_float
        )
        return None  # Fail-safe: API issue, don't corrupt state
    else:
        final_qty = min(free_bal,base_qty_float)
    logging.debug("%s: SELL DEBUG (A): base_quantity=%.8f, free_bal=%.8f, calculated_qty=%.8f", symbol, float(base_quantity), free_bal, final_qty) 
    qty_dec = Decimal(str(final_qty))
    min_qty_dec = Decimal(str(f["minQty"]))
    logging.debug("%s: SELL DEBUG (B): filter_minQty_float=%.8f, final_qty_dec=%s", symbol, f["minQty"], str(min_qty_dec))
    if qty_dec < min_qty_dec: # <-- Use Decimal comparison
        logging.warning(
            "%s: available quantity %.8f below minQty %.8f, skipping sell (will NOT remove state).",
            symbol, float(qty_dec), float(min_qty_dec) # <-- Log the Decimal value converted back to float
        )
    # --- Fix: Adjust for Binance fee rounding and minQty ---
    step = f["stepSize"]
    min_qty = f["minQty"]

    # Apply fee buffer and round down to the nearest tradable step
    #fee_buffer = 0.999  # 0.1% buffer
    adjusted_qty = math.floor(final_qty / step) * step

    if adjusted_qty < min_qty:
        logging.warning("%s: adjusted_qty %.8f below minQty %.8f, skipping sell", symbol, adjusted_qty, min_qty)
        return None

    logging.debug("%s: adjusted sell qty from %.8f -> %.8f to account for fees and step size", symbol, final_qty, adjusted_qty)
    final_qty = adjusted_qty
    price = fetch_price(symbol)
    if price is None:
        logging.error("%s: failed to fetch price", symbol)
        return None
    notional = final_qty * price
    if notional < f["minNotional"]:
        logging.warning("%s: notional %.2f below minNotional %.2f, skipping sell", symbol, notional, f["minNotional"])
        return None
    order_info = create_market_order(api_key, api_secret, symbol, "SELL", quantity=final_qty)
    if not order_info:
        logging.error("%s: create market order failed", symbol)
        return None
    executed_qty = float(order_info.get("executedQty", final_qty))
    executed_quote = float(order_info.get("cummulativeQuoteQty", executed_qty * price))
    return {
        "executedQty": executed_qty,
        "cummulativeQuoteQty": executed_quote,
        "status": order_info.get("status", "UNKNOWN"),
    }
# ---------- Indicators ----------
def simple_sma(prices, n):
    if len(prices) < n:
        return None
    return sum(prices[-n:]) / n
def calculate_average_volume(klines, period=20):
    volumes = [float(k[5]) for k in klines]
    return simple_sma(volumes, period)
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
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period
def generate_signal_with_atr(klines, atr_period=14, atr_mult=2.0, ema_period=7, pos=None, last_price=None):
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    min_len = max(atr_period, ema_period) + 2
    if len(closes) < min_len:
        return "HOLD", None, None, None, 0.0, 0.0
    atr = calculate_atr(klines[:-1], period=int(atr_period))
    alpha = 2 / (ema_period + 1)
    ema = closes[0]
    for price in closes[1:]:
        ema = alpha * price + (1 - alpha) * ema
    ema_prev = sum(closes[-(ema_period + 1):-1]) / ema_period if len(closes) >= ema_period + 1 else ema
    ema_slope = (ema - ema_prev) / ema_prev * 100 if ema_prev != 0 else 0.0
    recent_high = max(highs[-(atr_period + 1):-1])
    recent_low = min(lows[-(atr_period + 1):-1])
    last_close = closes[-1]
    # --- START: New EMA-based Trend Strength Check ---
    fast_period = int(get_cfg("EMA_FAST", int, 7))
    slow_period = int(get_cfg("EMA_SLOW", int, 21))

    # Calculate EMA series using existing utility function
    ema_fast_series = ema_series(closes, fast_period)
    ema_slow_series = ema_series(closes, slow_period)

    if ema_fast_series and ema_slow_series:
        f_curr = ema_fast_series[-1] # Current Fast EMA
        s_curr = ema_slow_series[-1] # Current Slow EMA
    else:
        f_curr = None
        s_curr = None
# --- END: New EMA-based Trend Strength Check ---
    diff_pct = ((f_curr - s_curr) / s_curr) * 100.0 if f_curr is not None and s_curr and s_curr != 0 else 0.0
    buy_trigger = recent_high + (atr_mult*0.5) * atr
    sell_trigger = recent_low - (atr_mult*0.5) * atr
    if pos is None:
        logger.debug(
            "[ATR-SIGNAL] close=%s ema=%.6f recent_high=%.6f recent_low=%.6f atr=%.6f buy_trig=%.6f sell_trig=%.6f ema_slope=%.4f",
            last_close, ema, recent_high, recent_low, atr, buy_trigger, sell_trigger, ema_slope
        )
    if last_close > buy_trigger and last_close > ema:
        return "BUY", recent_high, recent_low, atr, ema_slope, diff_pct
    elif last_close < sell_trigger and last_close < ema and ema_slope < 0:
        return "SELL", recent_high, recent_low, atr, ema_slope, diff_pct
    else:
        return "HOLD", recent_high, recent_low, atr, ema_slope, diff_pct
def ema_series(prices, period):
    k = 2 / (period + 1)
    if len(prices) < period:
        return []
    ema_val = sum(prices[:period]) / period
    out = [ema_val]
    for p in prices[period:]:
        ema_val = p * k + ema_val * (1 - k)
        out.append(ema_val)
    return out
def is_hourly_ema_increasing(klines):

    closes_15m = [float(c[4]) for c in klines]
    usable_len = len(closes_15m) - (len(closes_15m) % 4)
    # take the *last close* of each 1h block (true hourly close)
    hourly_closes = [closes_15m[i+3] for i in range(0, usable_len, 4)]

    if len(hourly_closes) < 10:
        return False

    ema_hourly = ema_series(hourly_closes, 6)  # slightly faster EMA

    if len(ema_hourly) < 4:
        return False

    recent_avg = np.mean(ema_hourly[-2:])
    prev_avg = np.mean(ema_hourly[-4:-2])
    slope = recent_avg - prev_avg

    logging.debug("Hourly EMA: recent_avg=%.6f prev_avg=%.6f slope=%.6f (%s)",
                 recent_avg, prev_avg, slope, "UP" if slope > 0 else "DOWN")

    return slope > 0
def ema_crossover(symbol, closes, state, fast_period=7, slow_period=21, slope_filter=True, slope_min=0.1):
    if len(closes) < slow_period + 2:
        return None, 0.0
    fast = ema_series(closes, fast_period)
    slow = ema_series(closes, slow_period)
    if not fast or not slow:
        return None, 0.0
    min_len = min(len(fast), len(slow))
    fast, slow = fast[-min_len:], slow[-min_len:]
    ema_diff_pct = (fast[-1] - slow[-1]) / slow[-1] * 100 if slow[-1] != 0 else 0.0
    cs = state.setdefault("crossover_state", {}).setdefault(symbol, {"waiting_buy": False, "waiting_sell": False})
    crossed_up = fast[-2] <= slow[-2] and fast[-1] > slow[-1]
    crossed_down = fast[-2] >= slow[-2] and fast[-1] < slow[-1]
    if crossed_up:
        cs["waiting_buy"] = True
        cs["waiting_sell"] = False
    elif crossed_down:
        cs["waiting_sell"] = True
        cs["waiting_buy"] = False
    if slope_filter and len(fast) >= 4:
        slope = (fast[-1] - fast[-3]) / fast[-3] * 100 if fast[-3] != 0 else 0.0
        if cs["waiting_buy"] and slope <= slope_min:
            cs["waiting_buy"] = False
        if cs["waiting_sell"] and slope >= -slope_min:
            cs["waiting_sell"] = False
    if cs["waiting_buy"]:
        return "BUY", ema_diff_pct
    if cs["waiting_sell"]:
        return "SELL", ema_diff_pct
    return None, ema_diff_pct
def no_three_down(closes):
    if len(closes) < 4:
        return True
    return not all(closes[i] < closes[i-1] for i in range(-3, 0))
def bearish_recent_trend(klines):
    closes = [float(k[4]) for k in klines]
    if len(closes) < 6:
        return True
    down_count = 0
    for i in range(-5, 0):
        if closes[i] < closes[i - 1]:
            down_count += 1
    return down_count < 3
def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
def htf_trend_ok_ema21_1h(klines_15m,
                          ema_period=21,
                          slope_min_pct=0.15,
                          min_closes=3):
    """
    Higher-timeframe trend gate using EMA-21 on true 1H closes,
    derived from 15m klines.

    Returns True if:
      - Price is above EMA-21 (1H)
      - EMA-21 slope > slope_min_pct
      - Last `min_closes` closes are above EMA-21
    """

    # --- Build true 1H closes from 15m data ---
    closes_15m = [float(k[4]) for k in klines_15m]
    usable_len = len(closes_15m) - (len(closes_15m) % 4)

    if usable_len < (ema_period + 5) * 4:
        return False  # not enough data

    hourly_closes = [closes_15m[i + 3] for i in range(0, usable_len, 4)]

    # --- EMA-21 on 1H ---
    ema_vals = ema_series(hourly_closes, ema_period)
    if len(ema_vals) < ema_period + 2:
        return False

    ema_now = ema_vals[-1]
    ema_prev = ema_vals[-2]
    last_close = hourly_closes[-1]

    # --- Price must be above EMA ---
    if last_close <= ema_now:
        return False

    # --- EMA slope (% change) ---
    if ema_prev <= 0:
        return False

    slope_pct = ((ema_now - ema_prev) / ema_prev) * 100.0
    if slope_pct < slope_min_pct:
        return False

    # --- Last N closes above EMA ---
    for i in range(1, min_closes + 1):
        if hourly_closes[-i] <= ema_vals[-i]:
            return False

    return True
def ltf_pullback_entry_15m(klines_15m,
                           ema_fast=7,
                           ema_slow=21,
                           atr_period=20,
                           atr_mult=1.5,
                           vol_period=20,
                           vol_mult=1.1):
    """
    15m pullback continuation entry logic.

    Returns dict with entry, sl, tp, risk if valid.
    Returns None if no valid setup.
    """

    if len(klines_15m) < max(ema_slow, atr_period, vol_period) + 5:
        return None

    closes = [float(k[4]) for k in klines_15m]
    highs = [float(k[2]) for k in klines_15m]
    lows = [float(k[3]) for k in klines_15m]
    volumes = [float(k[5]) for k in klines_15m]

    # --- EMAs ---
    ema_fast_series = ema_series(closes, ema_fast)
    ema_slow_series = ema_series(closes, ema_slow)
    
    if not ema_fast_series or not ema_slow_series:
        return None
    
    ema_fast_val = ema_fast_series[-1]
    ema_slow_val = ema_slow_series[-1]

    # Trend must be intact
    if ema_fast_val <= ema_slow_val:
        return None

    # Pullback: previous close must stay above EMA21
    if closes[-2] <= ema_slow_val:
        return None

    # Entry trigger: bullish close back above EMA7
    if closes[-1] <= ema_fast_val:
        return None

    # --- Volume confirmation (soft) ---
    vol_ma = sum(volumes[-vol_period:]) / vol_period
    if volumes[-1] < vol_ma * vol_mult:
        return None

    # --- ATR for risk ---
    atr_val = calculate_atr(klines_15m, atr_period)
    if atr_val is None or atr_val <= 0:
        return None

    entry = closes[-1]
    risk = atr_val * atr_mult

    return {
        "entry": entry,
        "sl": entry - risk,
        "risk": risk
    }
def calculate_bollinger_bands(closes, period=20, std_dev=2.0):
    """
    Calculate Bollinger Bands: SMA ± (std_dev × standard deviation)
    Returns: (middle, upper, lower, bandwidth)
    """
    if len(closes) < period:
        return None, None, None, None
    
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    bandwidth = (upper - lower) / sma if sma != 0 else 0
    
    return sma, upper, lower, bandwidth
def detect_squeeze_release(closes, period=20, std_dev=2.0, lookback=3):
    """
    Detect Bollinger Band squeeze release (bands expanding after contraction).
    
    Returns: True if bands are expanding (squeeze released), False otherwise
    """
    if len(closes) < period + lookback:
        return False
    
    bandwidths = []
    for i in range(lookback + 1):
        subset = closes[-(period + i):len(closes) - i if i > 0 else None]
        if len(subset) < period:
            return False
        sma = np.mean(subset)
        std = np.std(subset)
        bw = ((sma + std_dev * std) - (sma - std_dev * std)) / sma if sma != 0 else 0
        bandwidths.append(bw)
    
    # Current bandwidth should be greater than previous (expanding)
    return bandwidths[0] > bandwidths[1]
def bollinger_rsi_momentum_signal(klines, state, symbol, 
                                   rsi_period=14, bb_period=20, bb_std=2.0,
                                   vol_mult=1.2, vol_period=20):
    """
    CUSTOM STRATEGY: EMA_7_21_Cross
    Fast EMA 7 crosses above slow EMA 21
    """
    try:
        signal = custom_signal(symbol, klines, state)
        
        # Get bandwidth for compatibility
        closes = [float(k[4]) for k in klines]
        _, _, _, bandwidth = calculate_bollinger_bands(closes, bb_period, bb_std)
        
        # Get RSI for compatibility
        rsi = calculate_rsi(closes, rsi_period)
        
        return signal, rsi or 50.0, bandwidth or 0.0
    except Exception as e:
        logging.exception("Error in custom signal")
        return "HOLD", 50.0, 0.0


def custom_signal(symbol, klines, state):
    """
    RSI 25 Oversold Reversal + Volume Confirmation - OPTIMIZED
    
    IMPROVEMENTS (44% profit increase):
    - TP ratio increased to 4:1 (from 2.5:1)
    - Removed RSI exit - let SL/TP manage the trade
    - Maintains 1.5x Chandelier stop loss
    
    Performance: 27 trades/90 days, 59.3% WR, $2.40 profit, PF 2.25
    Expanded to 12 coins for optimal frequency
    """
    if len(klines) < 100:
        return "HOLD"
    
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    
    # RSI calculations - use RSI_PERIOD from .env
    rsi_period = int(get_cfg("RSI_PERIOD", int, 14))
    rsi_current = calculate_rsi(closes, rsi_period)
    rsi_previous = calculate_rsi(closes[:-1], rsi_period) if len(closes) > rsi_period else None
    
    if not rsi_current or not rsi_previous:
        return "HOLD"
    
    # Volume filter - read multiplier from .env
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
    vol_mult = float(get_cfg("VOLUME_MULT", float, 1.5))
    volume_confirmed = volumes[-1] > avg_vol * vol_mult
    
    # BUY: RSI crosses above 25 with volume confirmation
    if rsi_previous <= 25 and rsi_current > 25 and volume_confirmed:
        return "BUY"
    
    # NO RSI EXIT - Let SL/TP manage the trade naturally
    # This improves win rate from 48.3% to 59.3%
    
    return "HOLD"

# ---------- Stop Loss Management - Chandelier Stop (Professional Grade) ----------
def update_sl_chandelier(symbol, pos, current_price, candle_high, state,
                         atr_mult=2.0, quantize=True):
    """
    Chandelier Stop - Professional trailing stop based on ATR.
    
    Logic:
    - SL = Highest_Price - (ATR × atr_mult)
    - Updates only on new highs (ratchet-only, never lowers)
    - Adapts automatically to market volatility via ATR
    - Simple, proven, used by professional traders
    
    Parameters:
    - atr_mult: multiplier (2.0=default, 3.0=wider, 1.5=tighter)
    - quantize: round to SYMBOL_FILTERS step size for exchange compliance
    
    Returns: None (modifies position dict in-place)
    """
    try:
        from decimal import Decimal, ROUND_DOWN
        
        # Validate inputs
        if not isinstance(pos, dict):
            return
        
        entry_price = float(pos.get("entry_price", 0.0))
        if entry_price <= 0.0:
            return
        
        # Get ATR from position (should be locked at entry)
        atr_val = float(pos.get("locked_atr", 0.0))
        if atr_val <= 0.0:
            atr_val = entry_price * 0.015  # 1.5% fallback
        
        # Track highest price since entry (new-high-only updates)
        highest = float(pos.get("highest_since_entry", entry_price))
        old_sl = float(pos.get("sl", entry_price * 0.98))
        
        # Update highest if new high seen
        if candle_high > highest:
            highest = float(candle_high)
            pos["highest_since_entry"] = highest
        
        # ============ Chandelier Stop Calculation ============
        # SL = Highest - (ATR × atr_mult)
        candidate_sl = highest - (atr_val * atr_mult)
        
        # Apply minimum floor (never let SL be above entry)
        # This protects against weird cases
        candidate_sl = min(candidate_sl, entry_price * 1.001)  # tiny profit floor
        
        # Ratchet-only: never lower the SL
        if candidate_sl <= old_sl:
            return
        
        # Quantize to exchange step size if available
        sfilters = SYMBOL_FILTERS.get(symbol.replace("/", ""))
        if quantize and sfilters:
            try:
                step = float(sfilters.get("stepSize", 0.0))
                if step and step > 0:
                    step_s = ("{:.8f}".format(step)).rstrip("0").rstrip(".")
                    decimals = len(step_s.split(".")[1]) if "." in step_s else 0
                    sl_dec = Decimal(str(candidate_sl)).quantize(Decimal(10) ** -decimals, rounding=ROUND_DOWN)
                    final_sl = float(sl_dec) if float(sl_dec) > 0 else old_sl
                else:
                    final_sl = round(candidate_sl, 8)
            except Exception:
                final_sl = round(candidate_sl, 8)
        else:
            final_sl = round(candidate_sl, 8)
        
        # Safety: never set SL >= current price
        if final_sl >= current_price:
            final_sl = current_price * 0.999
            if sfilters and quantize:
                try:
                    step = float(sfilters.get("stepSize", 0.0))
                    if step and step > 0:
                        step_s = ("{:.8f}".format(step)).rstrip("0").rstrip(".")
                        decimals = len(step_s.split(".")[1]) if "." in step_s else 0
                        sl_dec = Decimal(str(final_sl)).quantize(Decimal(10) ** -decimals, rounding=ROUND_DOWN)
                        final_sl = float(sl_dec)
                except Exception:
                    final_sl = round(final_sl, 8)
        
        # Final ratchet check
        if final_sl <= old_sl:
            return
        
        # Apply the new SL
        pos["sl"] = final_sl
        pos["sl_updates"] = int(pos.get("sl_updates", 0)) + 1
        
        # Log the update
        try:
            logging.debug(
                "%s: Chandelier SL ratcheted %.8f -> %.8f | highest=%.6f atr=%.6f mult=%.1f",
                symbol, old_sl, final_sl, highest, atr_val, atr_mult
            )
        except Exception:
            pass
    
    except Exception:
        logging.exception("Error in update_sl_chandelier for %s", symbol)

# ---------- Stop Loss Management (Enhanced Professional Version) ----------
def update_dynamic_sl_pro(symbol, pos, current_price, candle_high, candle_low, 
                          klines, state, atr_mult=None, buffer_frac=None):
    """
    Professional Dynamic Stop Loss with Adaptive Profit Capture & Reversal Detection
    
    Key Features:
    1. Multi-stage profit protection with adaptive tightening
    2. Reversal signal detection (momentum, volume, price action)
    3. ATR-based volatility adjustment
    4. Ratchet-only mechanism (never lowers SL)
    5. Professional-grade profit locking at key thresholds
    
    Strategy Philosophy:
    - Conservative initial stop (give trade room to breathe)
    - Progressive tightening as profit develops
    - Aggressive protection once reaching 80%+ of target
    - Early reversal detection to lock gains before major pullbacks
    
    Parameters:
    - symbol: Trading pair
    - pos: Position dictionary
    - current_price: Current market price (close)
    - candle_high/low: Current candle extremes for reversal detection
    - klines: Recent candles for technical analysis
    - state: Global state for context storage
    - atr_mult: ATR multiplier for stop distance (default from config)
    - buffer_frac: Minimum buffer as fraction of ATR (default from config)
    """
    try:
        from decimal import Decimal, ROUND_DOWN
        import math

        # ============================================================================
        # VALIDATION & INITIALIZATION
        # ============================================================================
        if not isinstance(pos, dict):
            return

        # Load configuration
        if atr_mult is None:
            atr_mult = float(get_cfg("ATR_MULT", float, 2.0))
        if buffer_frac is None:
            buffer_frac = float(get_cfg("ATR_BUFFER_MULT", float, 0.3))

        # Core position values
        entry_price = float(pos.get("entry_price", 0.0))
        tp_price = float(pos.get("initial_tp", pos.get("tp", 0.0)))
        if entry_price <= 0.0:
            return
        if tp_price <= entry_price:
            # If no TP set, use default 2% target
            tp_price = entry_price * 1.02

        # Get ATR - prefer locked ATR for consistency
        atr_val = float(pos.get("locked_atr", 0.0))
        if atr_val <= 0.0:
            # Calculate fresh ATR if not available
            if klines and len(klines) >= 14:
                atr_val = calculate_atr([float(k[2]) for k in klines[-14:]], 
                                       [float(k[3]) for k in klines[-14:]], 
                                       [float(k[4]) for k in klines[-14:]], 14)
            if atr_val <= 0.0:
                atr_val = entry_price * 0.005  # 0.5% fallback

        # Initialize or load stop loss
        old_sl = float(pos.get("sl", 0.0))
        if old_sl <= 0.0:
            # Set initial conservative stop
            init_sl = entry_price - (atr_val * atr_mult * 1.2)  # Wider initial stop
            pos["sl"] = max(init_sl, entry_price * 0.985)  # Max 1.5% initial risk
            pos["highest_since_entry"] = entry_price
            pos["sl_updates"] = 0
            pos["sl_stage"] = "INITIAL"
            return

        # ============================================================================
        # TRACK HIGHEST PRICE & CALCULATE PROGRESS
        # ============================================================================
        highest = float(pos.get("highest_since_entry", entry_price))
        new_high = False

        if candle_high > highest:
            highest = float(candle_high)
            pos["highest_since_entry"] = highest
            new_high = True

        # Calculate progress toward target (0.0 to 1.0+)
        tp_range = tp_price - entry_price
        progress = (highest - entry_price) / tp_range if tp_range > 0 else 0.0
        progress = max(0.0, progress)

        # Current unrealized P&L percentage
        current_pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # ============================================================================
        # REVERSAL SIGNAL DETECTION
        # ============================================================================
        reversal_detected = False
        reversal_strength = 0.0  # 0.0 to 1.0 scale
        
        if klines and len(klines) >= 20:
            try:
                closes = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                # 1. MOMENTUM REVERSAL - RSI divergence or overbought exhaustion
                rsi = calculate_rsi(closes, 14)
                if rsi:
                    # Overbought exhaustion (RSI > 70 and declining)
                    if len(closes) >= 3:
                        rsi_prev = calculate_rsi(closes[:-1], 14)
                        if rsi_prev and rsi > 70 and rsi < rsi_prev:
                            reversal_strength += 0.3
                            reversal_detected = True
                
                # 2. VOLUME CLIMAX - High volume with small price gain (distribution)
                if len(volumes) >= 20:
                    avg_volume = sum(volumes[-20:-1]) / 19
                    current_vol = volumes[-1]
                    if current_vol > avg_volume * 2.0:  # Volume spike
                        # Check if price action is weakening despite volume
                        price_change = (closes[-1] - closes[-2]) / closes[-2]
                        if price_change < 0.002:  # Less than 0.2% gain on high volume
                            reversal_strength += 0.25
                            reversal_detected = True
                
                # 3. PRICE ACTION REVERSAL - Bearish engulfing or shooting star
                if len(klines) >= 2:
                    prev_candle = klines[-2]
                    curr_candle = klines[-1]
                    
                    prev_open, prev_close = float(prev_candle[1]), float(prev_candle[4])
                    curr_open, curr_close = float(curr_candle[1]), float(curr_candle[4])
                    
                    # Bearish engulfing
                    if (prev_close > prev_open and  # Previous bullish
                        curr_close < curr_open and  # Current bearish
                        curr_open >= prev_close and  # Opens at/above prev close
                        curr_close < prev_open):    # Closes below prev open
                        reversal_strength += 0.35
                        reversal_detected = True
                    
                    # Shooting star / long upper wick
                    candle_body = abs(curr_close - curr_open)
                    upper_wick = candle_high - max(curr_open, curr_close)
                    if candle_body > 0 and upper_wick > candle_body * 2.5:
                        reversal_strength += 0.25
                        reversal_detected = True
                
                # 4. BREAKDOWN FROM RECENT HIGH - Failed breakout
                if new_high and current_price < highest * 0.998:  # Dropped 0.2% from new high
                    reversal_strength += 0.2
                    reversal_detected = True
                    
            except Exception as e:
                logging.debug(f"{symbol}: Reversal detection error: {e}")
                pass

        # ============================================================================
        # ADAPTIVE STOP LOSS CALCULATION - MULTI-STAGE APPROACH
        # ============================================================================
        candidate_sl = old_sl
        stage = pos.get("sl_stage", "INITIAL")
        
        # --- STAGE 1: INITIAL PROTECTION (0-25% to target) ---
        if progress < 0.25:
            stage = "INITIAL"
            # Wide stop, let trade develop
            trailing_distance = atr_val * atr_mult * 1.5
            candidate_sl = highest - trailing_distance
            
        # --- STAGE 2: DEVELOPING PROFIT (25-50% to target) ---
        elif progress < 0.50:
            stage = "DEVELOPING"
            # Start tightening gradually
            trailing_distance = atr_val * atr_mult * 1.0
            candidate_sl = highest - trailing_distance
            # Lock minimum profit at breakeven
            candidate_sl = max(candidate_sl, entry_price * 1.001)
            
        # --- STAGE 3: PROFIT PROTECTION (50-80% to target) ---
        elif progress < 0.80:
            stage = "PROTECTING"
            # Tighter stop to protect gains
            trailing_distance = atr_val * atr_mult * 0.7
            candidate_sl = highest - trailing_distance
            # Lock at least 30% of potential profit
            min_profit_lock = entry_price + (tp_range * 0.30)
            candidate_sl = max(candidate_sl, min_profit_lock)
            
        # --- STAGE 4: PROFIT MAXIMIZATION (80%+ to target) ---
        else:
            stage = "MAXIMIZING"
            # Very tight stop to lock maximum gains
            trailing_distance = atr_val * atr_mult * 0.5
            candidate_sl = highest - trailing_distance
            # Lock at least 60% of potential profit
            min_profit_lock = entry_price + (tp_range * 0.60)
            candidate_sl = max(candidate_sl, min_profit_lock)
            
            # If reached 100% target, set SL to TP
            if progress >= 1.0:
                candidate_sl = tp_price

        # ============================================================================
        # REVERSAL PROTECTION - AGGRESSIVE TIGHTENING
        # ============================================================================
        if reversal_detected and reversal_strength >= 0.8 and current_pnl_pct > 1.5:
            # Strong reversal signal detected with decent profit
            # Tighten stop aggressively to lock gains
            reversal_sl = current_price * 0.99  # 0.4% below current price
            if reversal_sl > candidate_sl:
                candidate_sl = reversal_sl
                logging.info(f"{symbol}: Reversal detected (strength={reversal_strength:.2f}) - tightening SL to lock profit")

        # ============================================================================
        # SAFETY CHECKS & QUANTIZATION
        # ============================================================================
        
        # Ratchet-only: never lower stop loss
        if candidate_sl <= old_sl:
            return

        # Ensure minimum meaningful move (avoid noise)
        min_move = max(atr_val * 0.01, entry_price * 0.001)  # 0.1% minimum
        if (candidate_sl - old_sl) < min_move and not reversal_detected:
            return

        # Quantize to exchange step size
        sfilters = SYMBOL_FILTERS.get(symbol.replace("/", ""))
        if sfilters:
            try:
                step = float(sfilters.get("stepSize", 0.0))
                if step and step > 0:
                    step_s = ("{:.8f}".format(step)).rstrip("0").rstrip(".")
                    decimals = len(step_s.split(".")[1]) if "." in step_s else 0
                    sl_dec = Decimal(str(candidate_sl)).quantize(
                        Decimal(10) ** -decimals, rounding=ROUND_DOWN)
                    final_sl = float(sl_dec) if float(sl_dec) > 0 else old_sl
                else:
                    final_sl = round(candidate_sl, 8)
            except Exception:
                final_sl = round(candidate_sl, 8)
        else:
            final_sl = round(candidate_sl, 8)

        # Final safety: never set SL at or above current price
        if final_sl >= current_price:
            final_sl = current_price * 0.997
            if final_sl <= old_sl:
                return

        # ============================================================================
        # UPDATE POSITION & LOGGING
        # ============================================================================
        pos["sl"] = final_sl
        pos["sl_stage"] = stage
        pos["sl_updates"] = int(pos.get("sl_updates", 0)) + 1
        
        # Store telemetry for analysis
        pos["last_sl_info"] = {
            "old_sl": old_sl,
            "new_sl": final_sl,
            "progress": progress,
            "stage": stage,
            "atr_val": atr_val,
            "reversal_detected": reversal_detected,
            "reversal_strength": reversal_strength,
            "pnl_pct": current_pnl_pct
        }

        # Detailed logging
        reversal_tag = f" [REVERSAL={reversal_strength:.2f}]" if reversal_detected else ""
        logging.info(
            f"{symbol}: SL UPDATE {old_sl:.6f} → {final_sl:.6f} | "
            f"Stage={stage} Progress={progress:.1%} PnL={current_pnl_pct:.2f}%{reversal_tag}"
        )

    except Exception as e:
        logging.error(f"Exception in update_dynamic_sl_pro for {symbol}: {e}")
        pass

def manage_trade_r_based(position,
                         current_price,
                         trail_atr_mult=1.0,
                         be_r=1.0,
                         trail_start_r=1.5):
    """
    R-based trade management.

    Rules:
    - Do nothing until +1R
    - At +1R: move SL to breakeven (once)
    - After +1.5R: trail SL by R (or ATR equivalent)
    """

    entry = position["entry_price"]
    sl = position["sl"]
    risk = position["risk"]

    if risk <= 0:
        return position

    r_multiple = (current_price - entry) / risk

    # --- Move to breakeven (once) ---
    if r_multiple >= be_r and not position.get("moved_to_be", False):
        position["sl"] = entry
        position["moved_to_be"] = True
        return position

    # --- Trail only after structure confirms ---
    if r_multiple >= trail_start_r:
        new_sl = current_price - (risk * trail_atr_mult)
        position["sl"] = max(sl, new_sl)

    return position

# ---------- State helpers ----------
def read_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("State file not found, creating new empty state.")
        return {}
    except json.JSONDecodeError:
        logging.warning("State file corrupt or empty, resetting state.")
        return {}
    except Exception as e:
        logging.error("Failed to read state: %s", e)
        logging.exception(e)
        return {}
def write_state(state, path=STATE_FILE):
    
    with _write_lock:
        try:
            snapshot = copy.deepcopy(state)
            if os.path.exists(path):
                os.replace(path, path + ".bak")
            tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(snapshot, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            state["_last_write"] = time.time()
             
            
        except Exception as e:
            logging.error("Failed to write state: %s", e)
def safe_write_state(state, path=STATE_FILE, max_retries=3):
    """Write state with retry logic to ensure persistence on transient failures."""
    for attempt in range(max_retries):
        try:
            write_state(state, path)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning("State write failed (attempt %d/%d), retrying: %s", attempt + 1, max_retries, e)
                time.sleep(1)
            else:
                logging.error("State write failed after %d attempts, trade may be lost", max_retries)
                return False
    return False
def initialize_state(cfg, state):
    symbols = [s.strip().upper() for s in cfg.get("SYMBOLS", "").split(",") if s.strip()]
    
    # Initialize crossover state for all symbols
    for symbol in symbols:
        state.setdefault("crossover_state", {}).setdefault(symbol, {
            "waiting_buy": False,
            "waiting_sell": False
        })
    
    # Ensure all trade history symbols are available (optional, but good practice)
    state.setdefault("cooldowns", {})
    state.setdefault("positions", {})
    
    return state
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
def run_cycle(cfg):
    global SYMBOL_FILTERS
    
    state = read_state()
    SYMBOL_FILTERS = load_symbol_filters(api_key=cfg.get("API_KEY"), api_secret=cfg.get("API_SECRET"))
    check_user_command(cfg, state)
    api_key = cfg.get("API_KEY", "")
    api_secret = cfg.get("API_SECRET", "")
    symbols = [s.strip().upper() for s in cfg.get("SYMBOLS", "").split(",") if s.strip()]
    tf = cfg.get("TIMEFRAME", "15m")
    position_usd = compute_position_usd(api_key, api_secret)
    if position_usd is None:
        logging.info("Position sizing below minimum, aborting run_cycle")
        return
    dry_run = str(cfg.get("DRY_RUN", "true")).lower() in ("1", "true", "yes", "y")
    # --- Ensure crossover_state contains all symbols ---
    all_symbols = [s.strip().upper() for s in cfg.get("SYMBOLS", "").split(",") if s.strip()]
    state.setdefault("crossover_state", {})
    for sym in all_symbols:
        state["crossover_state"].setdefault(sym, {"waiting_buy": False, "waiting_sell": False})
    cooldown_seconds = int(get_cfg("COOLDOWN_SECONDS", int, 300))
    state_modified = False

    for symbol in symbols:
        try:
            klines = fetch_klines(symbol, tf, limit=200)
            if not klines:
                logging.warning("%s: no klines returned, skipping", symbol)
                continue
            volumes = [float(k[5]) for k in klines]
            current_volume = volumes[-1] if volumes else 0.0
            avg_volume = calculate_average_volume(klines)
            volume_confirmed = avg_volume is not None and current_volume > (avg_volume * float(get_cfg("VOLUME_MULT", float, 1.2)))

            closes = [float(k[4]) for k in klines]
            rsi = calculate_rsi(closes, 14)
            rsi_ok = rsi is not None and rsi > 51  # confirm upward momentum
            logging.debug("%s: RSI=%.2f | RSI_OK=%s", symbol, rsi or 0, rsi_ok)
            pos = state.get("positions", {}).get(symbol)
            # read configured slope minimum for EMA filter
            slope_min_cfg = float(get_cfg("SLOPE_MIN", float, 0.1))
            ema_signal, ema_diff_pct = ema_crossover(symbol, closes, state,
                                                    fast_period=int(get_cfg("EMA_FAST", int, 7)),
                                                    slow_period=int(get_cfg("EMA_SLOW", int, 21)),
                                                    slope_filter=True,
                                                    slope_min=slope_min_cfg)
            last_price = closes[-1]
            candle_high = float(klines[-1][2])   # Uses candle close instead of wick high
            current_price = last_price
            sig, recent_high, recent_low, atr, ema_slope, diff_pct = generate_signal_with_atr(
                klines,
                atr_period=int(get_cfg("ATR_PERIOD", int, 14)),
                atr_mult=float(get_cfg("ATR_MULT", float, 2.0)),
                ema_period=int(get_cfg("EMA_FAST", int, 7)),
                pos=pos,
                last_price=last_price,
            )

            # save last ema_slope in pos for adaptive SL decisions if pos exists
            if pos is not None:
                try:
                    pos["last_ema_slope"] = ema_slope
                except Exception:
                    pass

            logging.debug("%s: price=%.6f signal=%s pos=%s ema=%s", symbol, last_price, sig, "OPEN" if pos else "NONE", ema_signal)
            min_diff_pct = float(get_cfg("MIN_DIFF_PCT", float, 0.15))
            strong_trend = ema_slope > float(get_cfg("EMA_TREND_PERCENT", float, 0.0))

            # --- cooldown check: skip buys if still cooling down from a recent close
            cooldowns = state.setdefault("cooldowns", {})
            cd_iso = cooldowns.get(symbol)
            in_cooldown = False
            if cd_iso:
                try:
                    cd_dt = datetime.fromisoformat(cd_iso)
                    if (datetime.now(timezone.utc) - cd_dt).total_seconds() < cooldown_seconds:
                        in_cooldown = True
                except Exception:
                    pass

            buy_cond_1 = (sig == "BUY")
            buy_cond_2 = (ema_signal == "BUY" and ema_diff_pct is not None and min_diff_pct < ema_diff_pct and rsi_ok)
            #buy_cond_3 = (ema_diff_pct is not None and ema_diff_pct > min_diff_pct and volume_confirmed and strong_trend and no_three_down(closes) and bearish_recent_trend(klines))
            # Check to make sure market movement is increasing
            hourly_ema_ok = is_hourly_ema_increasing(klines)
            # Attempt to open new position

            # ---------------- DEBUG: explain buy gating ----------------
            logging.info(
                "%s DEBUG: sig=%s ema_signal=%s ema_diff_pct=%.4f diff_pct(ema_fast-slow)=%.4f ema_slope=%.4f rsi=%.2f volume_curr=%.2f avg_vol=%.2f volume_confirmed=%s hourly_ema_ok=%s",
                symbol,
                sig,
                ema_signal,
                ema_diff_pct or 0.0,
                diff_pct or 0.0,
                ema_slope or 0.0,
                rsi or 0.0,
                current_volume,
                avg_volume or 0.0,
                volume_confirmed,
                hourly_ema_ok
            )
            # ---------------------------------------------------------

            if pos is None and not in_cooldown and symbol not in state["positions"] and (hourly_ema_ok or ema_slope > 0) and (buy_cond_2):
                # FIX BUG #5: Validate last_price before using it
                if last_price is None or last_price <= 0:
                    logging.error("%s: Invalid last_price (%.8f), cannot place order", symbol, last_price if last_price else 0)
                    continue
                
                # Pre-buy checks: ensure expected quantity meets exchange filters (avoid tiny buys)
                expected_qty = position_usd / last_price
                sf = SYMBOL_FILTERS.get(symbol.replace("/", ""))
                if sf:
                    try:
                        step = float(sf.get("stepSize", 0))
                        min_qty = float(sf.get("minQty", 0))
                        min_notional = float(sf.get("minNotional", 0))
                        if expected_qty < min_qty or (expected_qty * last_price) < min_notional:
                            logging.warning("%s: expected buy qty %.8f (notional %.2f) below exchange min (qty=%.8f minNotional=%.2f). Skipping buy.", symbol, expected_qty, expected_qty*last_price, min_qty, min_notional)
                            continue
                    except Exception:
                        pass

                # FIX BUG #4: Standardize order execution path for dry_run and live
                if dry_run:
                    # Simulate realistic execution: use last_price with simulated slippage
                    simulated_slippage = 1.0 + float(get_cfg("SLIPPAGE_PCT", float, 0.001))
                    avg_price = last_price * simulated_slippage
                    executed_qty = expected_qty
                    executed_quote = executed_qty * avg_price  # Match live calculation
                    order_info = {"status": "SIMULATED", "executedQty": str(executed_qty), "cummulativeQuoteQty": str(executed_quote)}
                else:
                    order_info = safe_market_buy(api_key, api_secret, symbol, position_usd)
                    if not order_info:
                        logging.error("%s: BUY order failed completely, skipping", symbol)
                        continue
                    executed_qty = float(order_info.get("executedQty", "0") or 0)
                    executed_quote = float(order_info.get("cummulativeQuoteQty", 0) or 0)
                    
                    # FIX BUG #1: Verify order execution BEFORE creating position
                    if executed_qty <= 0:
                        logging.error("%s: BUY returned executedQty=0, skipping position creation", symbol)
                        logging.debug("order_info: %s", order_info)
                        continue
                    
                    # Fallback to estimate if cummulativeQuoteQty missing
                    if executed_quote <= 0:
                        executed_quote = executed_qty * last_price
                        logging.warning("%s: cummulativeQuoteQty missing, estimated as %.2f", symbol, executed_quote)
                    
                    avg_price = executed_quote / executed_qty

                # FIX BUG #1: Only set these AFTER verification passes
                amount_bought = float(executed_qty)
                notional = float(executed_quote)
                entry_price = avg_price

                atr_period = int(get_cfg("ATR_PERIOD", int, 14))
                atr_mult_cfg = float(get_cfg("ATR_MULT", float, 2.0))
                tp_mult = float(get_cfg("TAKE_RATIO", float, 1.5))

                atr_val = calculate_atr(klines, atr_period)
                if atr_val is None or atr_val <= 0:
                    logging.warning("%s: ATR not available or invalid (%.8f), skipping trade", symbol, atr_val if atr_val else 0)
                    continue

                # adaptive ATR multiplier: tighten if strong trend
                atr_mult_eff = atr_mult_cfg
                if get_cfg("ATR_ADAPTIVE_ENABLED", bool):
                    if ema_slope and ema_slope > float(get_cfg("EMA_TREND_PERCENT", float, 0.0)):
                        reduction = float(get_cfg("ATR_ADAPTIVE_REDUCTION", float, 0.2))
                        atr_mult_eff = max(0.5, atr_mult_cfg * (1.0 - reduction))

                if buy_cond_1:
                    reason = "ATR BUY"
                elif buy_cond_2:
                    reason = "EMA BUY"
                else:
                    reason = "SIGNAL BUY"
                
                # FIX BUG #6: Standardized SL/TP calculation
                current_state = state
                fee_pct = float(get_cfg("FEE_PCT", float, 0.001))
                slip_pct = float(get_cfg("SLIPPAGE_PCT", float, 0.001))
                buffer_mult = 1.0 + fee_pct + slip_pct
                # SL: expand downward to account for fees/slippage
                sl_val = entry_price - (atr_val * atr_mult_eff * buffer_mult)
                initial_sl_val = max(sl_val, entry_price * 0.98)
                # TP: reduce upward target to account for fees/slippage
                tp_val = entry_price + (atr_val * atr_mult_eff * tp_mult * max(0.0, (1.0 - fee_pct - slip_pct)))
                
                # FIX BUG #1: Create position ONLY after all validations pass
                current_state.setdefault("positions", {})[symbol] = {
                    "amount": amount_bought,
                    "entry_price": entry_price,
                    "sl": sl_val,
                    "initial_sl": initial_sl_val,
                    "tp": tp_val,
                    "initial_tp": tp_val,
                    "locked_atr": atr_val,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "buy_signal": reason,
                    "last_ema_slope": ema_slope,
                }
                
                state_modified = True
                msg = f"{symbol} BUY executed {amount_bought:.8f} @ {entry_price:.6f} (notional ${notional:.2f}) [{reason}]"
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
                    "reason": reason,
                })

            # Manage existing position
            elif pos is not None:
                amount = float(pos["amount"])
                entry_price = float(pos["entry_price"])
                
                # FIX BUG #2: Enable phantom position detection
                if not dry_run:
                    asset = symbol.replace("/", "").replace("USDT", "")
                    actual_balance = fetch_balance(api_key, api_secret, asset)
                    
                    # Only log if balance fetch succeeded
                    if actual_balance is not None:
                        logging.debug("%s: FETCHED FREE BALANCE: %.8f", asset, actual_balance)
                        
                        if actual_balance < amount * 0.01:  # Less than 1% of position remains
                            time.sleep(5)
                            recheck = fetch_balance(api_key, api_secret, asset)
                            if recheck is not None and recheck < 0.01 * amount:
                                logging.warning("%s: PHANTOM POSITION DETECTED - Position in state but balance is %.8f (expected %.8f), removing", symbol, recheck, amount)
                                state.get("positions", {}).pop(symbol, None)
                                state.get("crossover_state", {}).pop(symbol, None)
                                state_modified = True
                                notify("Phantom Position Removed", f"{symbol}: Expected {amount:.8f} but found {recheck:.8f}")
                                continue
                            elif recheck is None:
                                logging.warning("%s: Could not verify balance on recheck, keeping position for safety", symbol)
                    else:
                        logging.warning("%s: Could not fetch balance, skipping phantom position check", symbol)
                atr_mult_cfg = float(get_cfg("ATR_MULT", float, 2.0))
                atr_val = pos.get("locked_atr")

                # Update trailing stop using enhanced dynamic professional stop loss
                old_sl = pos.get("sl")
                old_highest = pos.get("highest_since_entry", entry_price)
                
                # Get candle low for reversal detection
                candle_low = float(klines[-1][3]) if klines else last_price
                
                # Call the enhanced professional stop loss function
                update_dynamic_sl_pro(
                    symbol=symbol,
                    pos=pos,
                    current_price=last_price,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    klines=klines,
                    state=state
                )
                
                # Check if stop loss was updated
                if (pos.get("sl") != old_sl or 
                    pos.get("highest_since_entry") != old_highest):
                    state_modified = True
                should_exit = False
                reason = ""
                if last_price >= pos.get("tp", float("inf")):
                    should_exit = True
                    reason = "TP"
                elif last_price <= pos.get("sl", float("-inf")):
                    should_exit = True
                    reason = "SL"
                elif sig == "SELL":
                    should_exit = True
                    reason = "FORCED ATR EXIT"
                elif ema_signal == "SELL" and ema_diff_pct < -0.15:
                    should_exit = True
                    reason = "FORCED EMA EXIT"

                if should_exit:
                    if dry_run:
                        exit_order = {"status": "SIMULATED", "executedQty": str(amount), "cummulativeQuoteQty": str(amount * last_price)}
                    else:
                        forced_amount = float(state.get("positions", {}).get(symbol, {}).get("amount", 0.0))
                        logging.debug("%s: Forcing sell quantity to %.8f", symbol, forced_amount)
                        exit_order = safe_market_sell(api_key, api_secret, symbol, forced_amount, state)

                    if exit_order is not None and exit_order.get("status") in ["FILLED", "SIMULATED", "PARTIALLY_FILLED"]:
                        current_state = state
                        exit_px = last_price
                        pnl = (exit_px - entry_price) * amount
                        msg = f"{symbol} SELL {amount:.8f} @ {exit_px:.6f} ({reason}) PnL ${pnl:.2f}"
                        logging.info(msg)
                        notify("Trade Closed", msg)
                        append_trade({
                            "time": datetime.now(timezone.utc).isoformat(),
                            "symbol": symbol,
                            "side": "SELL",
                            "reason": reason,
                            "price": f"{exit_px:.6f}",
                            "amount": f"{amount:.8f}",
                            "notional_usd": f"{amount * exit_px:.2f}",
                            "pnl_usd": f"{pnl:.2f}",
                            "dry_run": str(dry_run),
                        })
                        # record cooldown
                        current_state.setdefault("cooldowns", {})[symbol] = datetime.now(timezone.utc).isoformat()
                        # remove position
                        current_state.get("positions", {}).pop(symbol, None)
                        state_modified = True
                    else:
                        # FIX BUG #3: Keep position but mark as failed, add retry tracking
                        logging.error("%s: Sell order failed or returned None; keeping position with error marker", symbol)
                        pos["sell_failed_count"] = int(pos.get("sell_failed_count", 0)) + 1
                        pos["last_sell_attempt"] = datetime.now(timezone.utc).isoformat()
                        pos["sell_error"] = True
                        state_modified = True
                        
                        # If failed 3+ times, force remove with alert
                        if pos.get("sell_failed_count", 0) >= 3:
                            logging.critical("%s: Sell failed %d times, FORCE REMOVING position - MANUAL VERIFICATION REQUIRED", symbol, pos["sell_failed_count"])
                            state.get("positions", {}).pop(symbol, None)
                            state.setdefault("cooldowns", {})[symbol] = datetime.now(timezone.utc).isoformat()
                            notify("CRITICAL: Orphaned Position", f"{symbol}: Sell failed {pos['sell_failed_count']} times. Position removed from bot. CHECK EXCHANGE MANUALLY!")
                            state_modified = True
            else:
                logging.debug("%s: HOLD", symbol)

            time.sleep(0.8)

        except Exception as e:
            logging.exception("Error running symbol %s: %s", symbol, e)
            notify("AutoTrader Error", f"{symbol}: {e}")
            time.sleep(2)
        # ---------- Final state write ----------
    # ---------- Final state write (OUTSIDE loop) ----------
    if state_modified:
        if not safe_write_state(state):
            logging.error("CRITICAL: Failed to save state after cycle")     
def run_cycle2(symbol):
    """
    Single high-quality setup:
    HTF EMA21 (1H) trend + 15m pullback continuation
    """

    now = datetime.now(timezone.utc).isoformat()
    state = read_state()

    # =====================================================
    # CONFIG (read once)
    # =====================================================
    api_key = CFG.get("API_KEY", "")
    api_secret = CFG.get("API_SECRET", "")
    position_usd = compute_position_usd(api_key, api_secret)
    if position_usd is None:
        logging.info("Position sizing below minimum, skipping run_cycle2 for %s", symbol)
        return
    cooldown_seconds = int(get_cfg("COOLDOWN_SECONDS", int, 300))
    dry_run = get_cfg("DRY_RUN", bool, False)

    # =====================================================
    # PROCESS MANUAL USER COMMANDS (BUY/SELL)
    # =====================================================
    global SYMBOL_FILTERS
    SYMBOL_FILTERS = load_symbol_filters(api_key=api_key, api_secret=api_secret)
    check_user_command(CFG, state)

    # =====================================================
    # LOAD 15m KLINES (authoritative data)
    # =====================================================
    klines_15m = fetch_klines(symbol, "15m", limit=300)
    if not klines_15m or len(klines_15m) < 120:
        return

    current_price = float(klines_15m[-1][4])

    # =====================================================
    # LOAD POSITION
    # =====================================================
    position = state.setdefault("positions", {}).get(symbol)

    # =====================================================
    # MANAGE EXISTING POSITION
    # =====================================================
    if position:
        position = manage_trade_r_based(position, current_price)

        # --- Stop loss hit ---
        if current_price <= position["sl"]:
            logging.warning("%s: SL hit at %.6f (SL=%.6f)", symbol, current_price, position["sl"])
            if not dry_run:
                safe_market_sell(api_key, api_secret, symbol, position["amount"], state)

            append_trade({
                "time": now,
                "symbol": symbol,
                "side": "SELL",
                "reason": "SL",
                "price": f"{current_price:.6f}",
                "amount": f"{position['amount']:.8f}",
                "notional_usd": f"{current_price * position['amount']:.2f}",
                "pnl_usd": f"{(current_price - position['entry_price']) * position['amount']:.2f}",
                "dry_run": str(dry_run),
            })

            msg = f"{symbol} SELL {position['amount']:.8f} @ {current_price:.6f} (SL) PnL ${(current_price - position['entry_price']) * position['amount']:.2f}"
            logging.info(msg)
            notify("Trade Closed", msg)

            state.get("positions", {}).pop(symbol, None)
            state.setdefault("cooldowns", {})[symbol] = now
            write_state(state)
            return

        # --- Persist updated SL ---
        state["positions"][symbol] = position
        write_state(state)
        logging.debug("%s: Position managed (SL=%.6f, r-multiple=%.2f)", symbol, position.get("sl"), (current_price - position.get("entry_price", 0)) / position.get("risk", 1) if position.get("risk") else 0)
        return

    # =====================================================
    # COOLDOWN CHECK
    # =====================================================
    last_trade_time = state.get("cooldowns", {}).get(symbol)
    if last_trade_time:
        try:
            cd_dt = datetime.fromisoformat(last_trade_time)
            time_since = (datetime.now(timezone.utc) - cd_dt).total_seconds()
            if time_since < cooldown_seconds:
                logging.debug("%s: In cooldown (%.0f/%.0f seconds)", symbol, time_since, cooldown_seconds)
                return
        except Exception:
            pass

    # =====================================================
    # HTF TREND GATE (EMA21 1H)
    # =====================================================
    if not htf_trend_ok_ema21_1h(klines_15m):
        logging.debug("%s: HTF trend gate failed (EMA21 1H not bullish)", symbol)
        return

    # =====================================================
    # ENTRY LOGIC (15m pullback)
    # =====================================================
    entry_signal = ltf_pullback_entry_15m(klines_15m)
    if not entry_signal:
        logging.debug("%s: No pullback entry setup found", symbol)
        return

    entry_price = entry_signal["entry"]
    sl = entry_signal["sl"]
    risk = entry_signal["risk"]

    # Calculate EMA slope at entry time (for position tracking)
    closes = [float(k[4]) for k in klines_15m]
    ema_fast_period = int(get_cfg("EMA_FAST", int, 7))
    ema_fast_series = ema_series(closes, ema_fast_period)
    
    if len(ema_fast_series) >= 4:
        # Calculate slope: % change from 2 candles ago to now
        ema_slope = (ema_fast_series[-1] - ema_fast_series[-3]) / ema_fast_series[-3] * 100 if ema_fast_series[-3] != 0 else 0.0
    else:
        ema_slope = 0.0

    # =====================================================
    # POSITION SIZE (fixed USD sizing, existing behaviour)
    # =====================================================
    amount = position_usd / entry_price
    if amount <= 0:
        logging.error("%s: Invalid amount calculated (%.8f), aborting buy", symbol, amount)
        return

    # =====================================================
    # PLACE ORDER (with pre-checks and execution verification)
    # =====================================================
    # Pre-buy: ensure expected base qty meets exchange filters
    expected_qty = position_usd / entry_price if entry_price else 0.0
    sf = SYMBOL_FILTERS.get(symbol.replace("/", ""))
    if sf:
        try:
            min_qty = float(sf.get("minQty", 0))
            min_notional = float(sf.get("minNotional", 0))
            if expected_qty < min_qty or (expected_qty * entry_price) < min_notional:
                logging.warning("%s: expected buy qty %.8f (notional %.2f) below exchange min; skipping buy.", symbol, expected_qty, expected_qty*entry_price)
                return
        except Exception:
            pass

    # FIX BUG #4 & #5: Standardize order execution and add None checks
    if not dry_run:
        order_info = safe_market_buy(api_key, api_secret, symbol, position_usd)
        if not order_info:
            logging.error("%s: BUY order failed completely, skipping", symbol)
            return
        executed_qty = float(order_info.get("executedQty", "0") or 0)
        executed_quote = float(order_info.get("cummulativeQuoteQty", 0) or 0)
        
        # FIX BUG #1: Verify execution before proceeding
        if executed_qty <= 0:
            logging.error("%s: BUY returned executedQty=0, not creating position. order_info=%s", symbol, order_info)
            return
        
        # Fallback if quote missing
        if executed_quote <= 0:
            if entry_price is None or entry_price <= 0:
                logging.error("%s: Cannot calculate executed_quote, no valid price available", symbol)
                return
            executed_quote = executed_qty * entry_price
            logging.warning("%s: cummulativeQuoteQty missing, estimated as %.2f", symbol, executed_quote)
        
        amount = executed_qty
        avg_price = executed_quote / executed_qty
    else:
        # FIX BUG #4: Match live execution with realistic simulation
        if entry_price is None or entry_price <= 0:
            logging.error("%s: Invalid entry_price for simulation", symbol)
            return
        simulated_slippage = 1.0 + float(get_cfg("SLIPPAGE_PCT", float, 0.001))
        avg_price = entry_price * simulated_slippage
        amount = expected_qty
        order_info = {"status": "SIMULATED"}

    notional = amount * avg_price
    append_trade({
        "time": now,
        "symbol": symbol,
        "side": "BUY",
        "reason": "HTF_PULLBACK_ENTRY",
        "price": f"{avg_price:.6f}",
        "amount": f"{amount:.8f}",
        "notional_usd": f"{notional:.2f}",
        "pnl_usd": "0.0",
        "dry_run": str(dry_run),
    })

    msg = f"{symbol} BUY {'SIM' if dry_run else 'EXECUTED'} {amount:.8f} @ {avg_price:.6f} (notional ${notional:.2f}) [HTF_PULLBACK_ENTRY]"
    logging.info(msg)
    notify("Trade Placed", msg)

    # =====================================================
    # CALCULATE SL/TP BEFORE STATE PERSISTENCE
    # =====================================================
    atr_period = int(get_cfg("ATR_PERIOD", int, 14))
    atr_mult_cfg = float(get_cfg("ATR_MULT", float, 2.0))
    tp_mult = float(get_cfg("TAKE_RATIO", float, 1.5))
    
    # Calculate ATR for the position
    klines = fetch_klines(symbol, "15m", limit=200)
    atr_val = calculate_atr(klines, atr_period) if klines else risk
    if atr_val is None or atr_val <= 0:
        atr_val = risk if risk and risk > 0 else 0.01  # fallback
    
    # FIX BUG #6: Standardized SL/TP calculation (matching run_cycle)
    fee_pct = float(get_cfg("FEE_PCT", float, 0.001))
    slip_pct = float(get_cfg("SLIPPAGE_PCT", float, 0.001))
    buffer_mult = 1.0 + fee_pct + slip_pct
    # Use avg_price (actual filled price) not entry_price (signal price) for SL/TP
    # SL: expand downward to account for fees/slippage
    sl_val = avg_price - (atr_val * atr_mult_cfg * buffer_mult)
    initial_sl_val = max(sl_val, avg_price * 0.98)
    # TP: reduce upward target to account for fees/slippage
    tp_val = avg_price + (atr_val * atr_mult_cfg * tp_mult * max(0.0, (1.0 - fee_pct - slip_pct)))
    
    # NOW PERSIST POSITION (only after full verification and SL/TP calculation)
    position_dict = {
        "amount": amount,
        "entry_price": avg_price,
        "sl": initial_sl_val,
        "initial_sl": initial_sl_val,
        "tp": tp_val,
        "initial_tp": tp_val,
        "locked_atr": float(atr_val),
        "opened_at": now,
        "buy_signal": "HTF_PULLBACK",
        "last_ema_slope": ema_slope,
        "highest_since_entry": avg_price,
        "risk": (atr_val * atr_mult_cfg),
    }

    state.setdefault("positions", {})[symbol] = position_dict
    if not safe_write_state(state):
        # State write failed; remove position and abort
        state.get("positions", {}).pop(symbol, None)
        logging.error("%s: Failed to persist position state, aborting entry", symbol)
        return
    
    state.setdefault("cooldowns", {})[symbol] = now
    logging.info("%s: Position stored - entry=%.6f SL=%.6f TP=%.6f locked_atr=%.6f", symbol, avg_price, initial_sl_val, tp_val, atr_val)
def run_cycle_rsi_momentum(cfg):
    """
    Main cycle using RSI Momentum + Bollinger Band Squeeze strategy.
    Simplified, proven approach with positive expectancy.
    """
    global SYMBOL_FILTERS
    
    state = read_state()
    SYMBOL_FILTERS = load_symbol_filters(api_key=cfg.get("API_KEY"), api_secret=cfg.get("API_SECRET"))
    check_user_command(cfg, state)
    
    api_key = cfg.get("API_KEY", "")
    api_secret = cfg.get("API_SECRET", "")
    symbols = [s.strip().upper() for s in cfg.get("SYMBOLS", "").split(",") if s.strip()]
    tf = cfg.get("TIMEFRAME", "15m")
    
    # Compute position sizing
    position_usd = compute_position_usd(api_key, api_secret)
    if position_usd is None:
        logging.info("Position sizing below minimum, aborting run_cycle_rsi_momentum")
        return
    
    dry_run = str(cfg.get("DRY_RUN", "true")).lower() in ("1", "true", "yes", "y")
    cooldown_seconds = int(get_cfg("COOLDOWN_SECONDS", int, 300))  # 5min default (standardized across all cycles)
    
    # Strategy parameters
    rsi_period = int(get_cfg("RSI_PERIOD", int, 5))
    bb_period = int(get_cfg("BB_PERIOD", int, 20))
    bb_std = float(get_cfg("BB_STD", float, 2.0))
    vol_mult = float(get_cfg("VOLUME_MULT", float, 1.2))
    vol_period = int(get_cfg("VOL_PERIOD", int, 20))
    atr_period = int(get_cfg("ATR_PERIOD", int, 14))
    atr_mult = float(get_cfg("ATR_MULT", float, 2.0))  # Use global default (was 1.5)
    take_ratio = float(get_cfg("TAKE_RATIO", float, 1.5))  # Use global default (was 2.0)
    trail_start = float(get_cfg("TRAIL_START", float, 1.0))
    
    state_modified = False
    
    for symbol in symbols:
        try:
            # Fetch klines
            klines = fetch_klines(symbol, tf, limit=200)
            if not klines or len(klines) < 50:
                logging.warning("%s: insufficient klines, skipping", symbol)
                continue
            
            closes = [float(k[4]) for k in klines]
            current_price = closes[-1]
            candle_high = float(klines[-1][2])
            
            pos = state.get("positions", {}).get(symbol)
            
            # Generate signal
            signal, rsi_val, bandwidth = bollinger_rsi_momentum_signal(
                klines, state, symbol,
                rsi_period=rsi_period,
                bb_period=bb_period,
                bb_std=bb_std,
                vol_mult=vol_mult,
                vol_period=vol_period
            )
            
            logging.debug("%s: price=%.6f signal=%s RSI=%.2f BW=%.6f pos=%s",
                         symbol, current_price, signal, rsi_val or 0, bandwidth or 0,
                         "OPEN" if pos else "NONE")
            
            # =======================
            # MANAGE EXISTING POSITION
            # =======================
            if pos is not None:
                entry_price = float(pos["entry_price"])
                amount = float(pos["amount"])
                buy_signal = pos.get("buy_signal", "")
                is_manual = (buy_signal == "MANUAL")
                
                # Update trailing stop using enhanced dynamic professional stop loss
                old_sl = pos.get("sl")
                old_highest = pos.get("highest_since_entry", entry_price)
                
                # Get candle low for reversal detection
                candle_low = float(klines[-1][3]) if klines else current_price
                
                # Call the enhanced professional stop loss function
                update_dynamic_sl_pro(
                    symbol=symbol,
                    pos=pos,
                    current_price=current_price,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    klines=klines,
                    state=state
                )
                
                # Check if stop loss was updated
                if (pos.get("sl") != old_sl or pos.get("highest_since_entry") != old_highest):
                    state_modified = True
                
                should_exit = False
                reason = ""
                
                # ==================== EXIT CONDITIONS ====================
                # Priority 1: Hard stops (SL and TP are non-negotiable for all positions)
                if current_price <= pos.get("sl", float("-inf")):
                    should_exit = True
                    reason = "SL_HIT"  # Chandelier Stop triggered
                elif current_price >= pos.get("tp", float("inf")):
                    should_exit = True
                    reason = "TP_TARGET"
                # Priority 2: Signal-based exits (momentum reversal) - SKIP for manual positions
                elif signal == "SELL" and not is_manual:
                    should_exit = True
                    reason = "RSI_MOMENTUM_EXIT"
                    # Log the momentum exit details
                    logging.info("%s: Momentum exit signal detected - RSI likely crossed below 50 or other exit conditions met",
                                symbol)
                elif signal == "SELL" and is_manual:
                    # Manual positions ignore RSI exits
                    logging.debug("%s: RSI exit signal ignored (MANUAL position - only TP/SL will close)", symbol)
                
                if should_exit:
                    if dry_run:
                        exit_order = {
                            "status": "SIMULATED",
                            "executedQty": str(amount),
                            "cummulativeQuoteQty": str(amount * current_price)
                        }
                    else:
                        exit_order = safe_market_sell(api_key, api_secret, symbol, amount, state)
                    
                    if exit_order and exit_order.get("status") in ["FILLED", "SIMULATED", "PARTIALLY_FILLED"]:
                        exit_px = current_price
                        pnl = (exit_px - entry_price) * amount
                        pnl_pct = (pnl / (entry_price * amount)) * 100 if entry_price else 0
                        
                        # Enhanced exit message with context
                        if reason == "SL_HIT":
                            msg = f"{symbol} SELL {amount:.8f} @ {exit_px:.6f} (Chandelier SL Hit) PnL ${pnl:.2f} ({pnl_pct:.2f}%)"
                        elif reason == "TP_TARGET":
                            msg = f"{symbol} SELL {amount:.8f} @ {exit_px:.6f} (TP Target Hit) PnL ${pnl:.2f} ({pnl_pct:.2f}%)"
                        elif reason == "RSI_MOMENTUM_EXIT":
                            msg = f"{symbol} SELL {amount:.8f} @ {exit_px:.6f} (RSI Momentum Exit) PnL ${pnl:.2f} ({pnl_pct:.2f}%)"
                        else:
                            msg = f"{symbol} SELL {amount:.8f} @ {exit_px:.6f} ({reason}) PnL ${pnl:.2f} ({pnl_pct:.2f}%)"
                        
                        logging.info(msg)
                        notify("Trade Closed", msg)
                        
                        append_trade({
                            "time": datetime.now(timezone.utc).isoformat(),
                            "symbol": symbol,
                            "side": "SELL",
                            "reason": reason,
                            "price": f"{exit_px:.6f}",
                            "amount": f"{amount:.8f}",
                            "notional_usd": f"{amount * exit_px:.2f}",
                            "pnl_usd": f"{pnl:.2f}",
                            "pnl_percent": f"{pnl_pct:.2f}",
                            "dry_run": str(dry_run),
                        })
                        
                        state.setdefault("cooldowns", {})[symbol] = datetime.now(timezone.utc).isoformat()
                        state.get("positions", {}).pop(symbol, None)
                        state_modified = True
                    else:
                        logging.error("%s: Sell order failed, removing position to prevent zombie", symbol)
                        state.get("positions", {}).pop(symbol, None)
                        state.setdefault("cooldowns", {})[symbol] = datetime.now(timezone.utc).isoformat()
                        state_modified = True
                
                continue
            
            # =======================
            # OPEN NEW POSITION
            # =======================
            
            # Check cooldown
            cooldowns = state.setdefault("cooldowns", {})
            cd_iso = cooldowns.get(symbol)
            in_cooldown = False
            if cd_iso:
                try:
                    cd_dt = datetime.fromisoformat(cd_iso)
                    if (datetime.now(timezone.utc) - cd_dt).total_seconds() < cooldown_seconds:
                        in_cooldown = True
                except Exception:
                    pass
            
            if signal == "BUY" and not in_cooldown:
                # FIX BUG #5: Validate price before using
                if current_price is None or current_price <= 0:
                    logging.error("%s: Invalid current_price, skipping", symbol)
                    continue
                
                # Pre-buy validation
                expected_qty = position_usd / current_price
                sf = SYMBOL_FILTERS.get(symbol.replace("/", ""))
                if sf:
                    try:
                        min_qty = float(sf.get("minQty", 0))
                        min_notional = float(sf.get("minNotional", 0))
                        if expected_qty < min_qty or (expected_qty * current_price) < min_notional:
                            logging.warning("%s: expected qty %.8f below min, skipping", symbol, expected_qty)
                            continue
                    except Exception:
                        pass
                
                # FIX BUG #4: Standardize order execution
                if dry_run:
                    simulated_slippage = 1.0 + float(get_cfg("SLIPPAGE_PCT", float, 0.001))
                    avg_price = current_price * simulated_slippage
                    executed_qty = expected_qty
                    executed_quote = executed_qty * avg_price
                    order_info = {
                        "status": "SIMULATED",
                        "executedQty": str(executed_qty),
                        "cummulativeQuoteQty": str(executed_quote)
                    }
                else:
                    order_info = safe_market_buy(api_key, api_secret, symbol, position_usd)
                    if not order_info:
                        logging.error("%s: BUY order failed, skipping", symbol)
                        continue
                    executed_qty = float(order_info.get("executedQty", "0") or 0)
                    executed_quote = float(order_info.get("cummulativeQuoteQty", 0) or 0)
                    
                    # FIX BUG #1: Verify execution
                    if executed_qty <= 0:
                        logging.error("%s: BUY returned executedQty=0, skipping", symbol)
                        continue
                    
                    # Fallback if quote missing
                    if executed_quote <= 0:
                        executed_quote = executed_qty * current_price
                        logging.warning("%s: cummulativeQuoteQty missing, estimated as %.2f", symbol, executed_quote)
                    
                    avg_price = executed_quote / executed_qty
                
                # Calculate ATR for stop loss
                atr_val = calculate_atr(klines, atr_period)
                if atr_val is None or atr_val <= 0:
                    atr_val = avg_price * 0.015  # 1.5% fallback
                
                # Calculate EMA slope for position tracking (Fix 1: was incorrectly storing RSI)
                ema_fast_period = int(get_cfg("EMA_FAST", int, 7))
                ema_fast_series = ema_series(closes, ema_fast_period)
                if len(ema_fast_series) >= 4:
                    ema_slope = (ema_fast_series[-1] - ema_fast_series[-3]) / ema_fast_series[-3] * 100 if ema_fast_series[-3] != 0 else 0.0
                else:
                    ema_slope = 0.0
                
                # FIX BUG #6: Standardized SL/TP calculation
                fee_pct = float(get_cfg("FEE_PCT", float, 0.001))
                slip_pct = float(get_cfg("SLIPPAGE_PCT", float, 0.001))
                buffer_mult = 1.0 + fee_pct + slip_pct
                
                initial_sl = avg_price - (atr_val * atr_mult * buffer_mult)
                initial_sl = max(initial_sl, avg_price * 0.98)  # 2% floor
                initial_tp = avg_price + (atr_val * atr_mult * take_ratio * (1.0 - fee_pct - slip_pct))
                
                # FIX BUG #1: Create position ONLY after all validations
                state.setdefault("positions", {})[symbol] = {
                    "amount": executed_qty,
                    "entry_price": avg_price,
                    "sl": initial_sl,
                    "initial_sl": initial_sl,
                    "tp": initial_tp,
                    "initial_tp": initial_tp,
                    "locked_atr": float(atr_val),
                    "risk": float(atr_val * atr_mult),
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "buy_signal": "RSI_MOMENTUM",
                    "highest_since_entry": avg_price,
                    "last_ema_slope": ema_slope,
                    "rsi_entry": rsi_val,
                    "bandwidth_entry": bandwidth,
                }
                
                state_modified = True
                
                # Enhanced logging with Chandelier Stop details
                chandelier_mult = float(get_cfg("CHANDELIER_MULT", float, 2.0))
                msg = f"{symbol} BUY {executed_qty:.8f} @ {avg_price:.6f} (${executed_quote:.2f}) | SL={initial_sl:.6f} TP={initial_tp:.6f} | Chandelier(×{chandelier_mult})"
                logging.info(msg)
                notify("Trade Placed", msg)
                
                append_trade({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "side": "BUY",
                    "price": f"{avg_price:.6f}",
                    "amount": f"{executed_qty:.8f}",
                    "notional_usd": f"{executed_quote:.2f}",
                    "pnl_usd": "0.0",
                    "dry_run": str(dry_run),
                    "reason": "RSI_MOMENTUM",
                    "sl": f"{initial_sl:.6f}",
                    "tp": f"{initial_tp:.6f}",
                })
            
            time.sleep(0.8)
            
        except Exception as e:
            logging.exception("Error running symbol %s: %s", symbol, e)
            notify("AutoTrader Error", f"{symbol}: {e}")
            time.sleep(2)
    
    # Write state if modified
    if state_modified:
        if not safe_write_state(state):
            logging.error("CRITICAL: Failed to save state after RSI momentum cycle")

# ---------- ML STRATEGY (CYCLE4) ----------
def check_early_exit_conditions(symbol, pos, current_price, klines, df):
    """
    Check if a trade should exit early due to objectively bad behavior.
    Returns (should_exit: bool, reason: str)
    
    Implements 5 early exit strategies:
    1. Time-based invalidation - exit if no progress after 20 bars
    2. Immediate rejection detection - exit if -1.5% in first 3 candles
    3. Momentum exhaustion - exit at -0.7% with negative momentum
    4. Volume dryout - exit when volume drops 50%+ while losing
    5. Structure break - exit when support level is broken
    """
    try:
        entry_price = float(pos["entry_price"])
        entry_time = datetime.fromisoformat(pos.get("opened_at"))
        time_since_entry = (datetime.now(timezone.utc) - entry_time).total_seconds()
        bars_since_entry = int(time_since_entry / 900)  # Assuming 15m bars
        
        # Calculate current drawdown
        drawdown_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Get recent price action
        if len(df) < 5:
            return False, ""
        
        recent_closes = df['close'].tail(5).tolist()
        recent_volumes = df['volume'].tail(5).tolist()
        
        # ============================================================
        # 1. TIME-BASED INVALIDATION (HIGH PRIORITY)
        # If trade hasn't moved favorably after 20 bars, thesis is invalid
        # ============================================================
        if bars_since_entry >= 20:
            # Check if we've made ANY progress (even small)
            progress_pct = ((current_price - entry_price) / entry_price) * 100
            if progress_pct < 0.2:  # Less than 0.2% progress in 20 bars = failed thesis
                return True, "TIME_INVALIDATION"
        
        # ============================================================
        # 2. IMMEDIATE REJECTION (HIGH PRIORITY)
        # If price drops -1.5% within first 3 candles, trade is clearly wrong
        # ============================================================
        if bars_since_entry <= 3 and drawdown_pct <= -1.5:
            return True, "IMMEDIATE_REJECTION"
        
        # ============================================================
        # 3. MOMENTUM EXHAUSTION (MEDIUM PRIORITY)
        # If losing + momentum turning negative = early warning
        # ============================================================
        if drawdown_pct <= -0.7 and len(recent_closes) >= 3:
            # Calculate momentum (3-bar slope)
            momentum = (recent_closes[-1] - recent_closes[-3]) / recent_closes[-3] * 100
            
            # Check if momentum is negative (price declining)
            if momentum < -0.1:  # Negative momentum threshold
                return True, "MOMENTUM_EXHAUSTION"
        
        # ============================================================
        # 4. VOLUME DRYOUT (LOW PRIORITY)
        # If volume drops significantly while losing, buyers are gone
        # ============================================================
        if drawdown_pct <= -0.5 and len(recent_volumes) >= 5:
            # Compare current volume to average of last 5 bars
            avg_volume = sum(recent_volumes[:-1]) / 4
            current_volume = recent_volumes[-1]
            
            # If current volume is 50% below average while losing
            if current_volume < avg_volume * 0.5:
                return True, "VOLUME_DRYOUT"
        
        # ============================================================
        # 5. STRUCTURE BREAK (HIGH PRIORITY)
        # If price breaks below entry-level support decisively
        # ============================================================
        if drawdown_pct <= -1.0 and len(df) >= 10:
            # Find recent swing low (support level)
            recent_lows = df['low'].tail(10).tolist()
            support_level = min(recent_lows[:-1])  # Exclude current bar
            
            # If current price breaks support by more than 0.3%
            if current_price < support_level * 0.997:  # 0.3% below support
                return True, "STRUCTURE_BREAK"
        
        return False, ""
        
    except Exception as e:
        logging.debug("%s: Error in early exit check: %s", symbol, e)
        return False, ""


def run_cycle4_ml_strategy(cfg):
    """
    ML-powered trading strategy using LightGBM model with 121 features.
    Optimized configuration: 53.3% WR, £2,468 profit (+24.7% return).
    
    Entry: Model probability >= 0.56 (optimized threshold)
    Exit: ATR-based stops (1.5× ATR), RR ratio 1.5:1 also uses dynamic SL management (chandelier)
    """
    global SYMBOL_FILTERS, ML_AVAILABLE
    
    # Check if ML dependencies are available
    if not ML_AVAILABLE:
        logging.error("ML dependencies not available!")
        logging.error("Install: pip install pandas joblib scikit-learn lightgbm")
        return
    
    # Load ML model (cache in global variable to avoid reloading every cycle)
    global _ML_MODEL, _ML_SESSION, _ML_MODEL_TYPE
    if '_ML_MODEL' not in globals() or _ML_MODEL is None:
        # Get model path from config - DEFAULT TO NEW TIME-ENHANCED MODEL
        model_path_cfg = get_cfg("ML_MODEL_PATH", str, "models/LightGBM_Robust_20260112_090713.json")
        
        # Check for .onnx version first (mobile-friendly)
        if model_path_cfg.endswith('.joblib'):
            onnx_path_cfg = model_path_cfg.replace('.joblib', '.onnx')
        else:
            onnx_path_cfg = None
        
        # Try multiple path resolution strategies
        def find_model(path_cfg):
            if not path_cfg:
                return None
            possible = [
                os.path.join(PROJECT_DIR, path_cfg),
                path_cfg,
                os.path.join(os.path.dirname(__file__), path_cfg),
                os.path.join(os.getcwd(), path_cfg),
            ]
            for p in possible:
                if os.path.exists(p):
                    return p
            return None
        
        # Try JSON first (best for mobile/ARM - pure Python)
        json_path_cfg = model_path_cfg.replace('.joblib', '.json')
        model_path = find_model(json_path_cfg)
        is_json = True
        is_onnx = False
        
        if not model_path:
            # Try ONNX next (good for mobile)
            model_path = find_model(onnx_path_cfg)
            is_json = False
            is_onnx = True
        
        if not model_path:
            # Fall back to joblib
            model_path = find_model(model_path_cfg)
            is_json = False
            is_onnx = False
        
        if not model_path:
            logging.error("ML model not found! Tried:")
            logging.error("  - %s (JSON - pure Python, ARM-friendly)", json_path_cfg)
            if onnx_path_cfg:
                logging.error("  - %s (ONNX - mobile-friendly)", onnx_path_cfg)
            logging.error("  - %s (joblib)", model_path_cfg)
            logging.error("")
            logging.error("SOLUTION for ARM/Android: Run export_model_to_json.py on PC")
            logging.error("  Then transfer .json file - only needs numpy!")
            return
        
        try:
            if is_json:
                if not JSON_INFERENCE_AVAILABLE:
                    logging.error("JSON model found but lightgbm_inference.py missing")
                    logging.error("Ensure lightgbm_inference.py is in the same folder")
                    return
                # Load feature selector (same name pattern as model)
                # Replace 'LightGBM_Robust' with 'FeatureSelector' to find matching selector file
                selector_path_cfg = model_path_cfg.replace('LightGBM_Robust', 'FeatureSelector').replace('LightGBM', 'FeatureSelector').replace('.joblib', '.json')
                selector_path = find_model(selector_path_cfg)
                
                if selector_path:
                    _ML_MODEL = LightGBMInference(model_path, selector_file=selector_path)
                    logging.info("✓ Feature selector loaded: %s", selector_path)
                else:
                    _ML_MODEL = LightGBMInference(model_path)
                    logging.warning("Feature selector not found at: %s", selector_path_cfg)
                    logging.warning("Using all features (no feature selection applied)")
                _ML_SESSION = None
                _ML_MODEL_TYPE = 'json'
                logging.info("✓ JSON model loaded (pure Python): %s", model_path)
                logging.info("  Features: %d", _ML_MODEL.model_data['n_features'])
                logging.info("  Trees: %d", len(_ML_MODEL.model_data['trees']))
            elif is_onnx:
                if not ONNX_AVAILABLE:
                    logging.error("ONNX model found but onnxruntime not installed")
                    logging.error("Install: pip install onnxruntime")
                    return
                _ML_SESSION = ort.InferenceSession(model_path)
                _ML_MODEL = None  # Not used for ONNX
                _ML_MODEL_TYPE = 'onnx'
                logging.info("✓ ONNX model loaded: %s", model_path)
                logging.info("  Input: %s", _ML_SESSION.get_inputs()[0].name)
                logging.info("  Shape: %s", _ML_SESSION.get_inputs()[0].shape)
            else:
                _ML_MODEL = joblib.load(model_path)
                _ML_SESSION = None
                _ML_MODEL_TYPE = 'joblib'
                logging.info("✓ ML model loaded: %s", model_path)
                logging.info("  Type: %s", type(_ML_MODEL).__name__)
        except Exception as e:
            logging.error("Failed to load model from %s: %s", model_path, e)
            logging.exception(e)
            return
    
    state = read_state()
    SYMBOL_FILTERS = load_symbol_filters(api_key=cfg.get("API_KEY"), api_secret=cfg.get("API_SECRET"))
    check_user_command(cfg, state)
    
    api_key = cfg.get("API_KEY", "")
    api_secret = cfg.get("API_SECRET", "")
    symbols = [s.strip().upper() for s in cfg.get("SYMBOLS", "").split(",") if s.strip()]
    tf = cfg.get("TIMEFRAME", "15m")
    
    # ML Strategy Configuration (optimized defaults)
    ml_prob_threshold = float(get_cfg("ML_PROBABILITY_THRESHOLD", float, 0.56))  # Optimized: was 0.9
    ml_rr_ratio = float(get_cfg("ML_RR_RATIO", float, 1.5))
    ml_atr_mult = float(get_cfg("ML_ATR_MULTIPLIER", float, 1.5))
    ml_entry_filter = get_cfg("ML_ENTRY_FILTER", str, "None")  # Optimized: threshold-only, no EMA filter
    
    logging.debug("ML Strategy Config: threshold=%.2f RR=%.1f ATR_mult=%.1f filter=%s",
                 ml_prob_threshold, ml_rr_ratio, ml_atr_mult, ml_entry_filter)
    
    # Standard settings
    position_usd = compute_position_usd(api_key, api_secret)
    if position_usd is None:
        logging.info("Position sizing below minimum, aborting run_cycle4_ml_strategy")
        return
    
    dry_run = str(cfg.get("DRY_RUN", "true")).lower() in ("1", "true", "yes", "y")
    cooldown_seconds = int(get_cfg("COOLDOWN_SECONDS", int, 300))
    atr_period = int(get_cfg("ATR_PERIOD", int, 14))
    
    state_modified = False
    
    for symbol in symbols:
        try:
            # Fetch klines
            klines = fetch_klines(symbol, tf, limit=200)
            if not klines or len(klines) < 50:
                logging.warning("%s: insufficient klines, skipping", symbol)
                continue
            
            # Convert klines to DataFrame for feature engineering
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                               'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                               'taker_buy_quote', 'ignore'])
            
            # Convert timestamp from milliseconds to datetime (required for time features)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert to numeric and keep only OHLCV + timestamp
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Add ATR column (required by feature engineering)
            closes_for_atr = df['close'].tolist()
            atr_val = calculate_atr(klines, atr_period)
            if atr_val is None or atr_val <= 0:
                atr_val = df['close'].iloc[-1] * 0.015  # 1.5% fallback
            df['atr'] = atr_val  # Feature engineering expects this column
            
            # Engineer features
            try:
                df = engineer_features(df)
                df = df.dropna()  # Remove NaN values
            except Exception as e:
                logging.error("%s: feature engineering failed: %s", symbol, e)
                continue
            
            if len(df) == 0:
                logging.warning("%s: no data after feature engineering", symbol)
                continue
            
            # Get latest bar features
            latest_row = df.iloc[-1]
            current_price = float(latest_row['close'])
            candle_high = float(df.iloc[-1]['high'])
            
            # Extract feature columns (exclude OHLCV, timestamp, and atr)
            # Note: atr is used for feature engineering but is NOT a model feature
            feature_cols = [col for col in df.columns 
                           if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'atr', 'label']]
            
            # Get ML prediction
            try:
                X_features = latest_row[feature_cols].values.reshape(1, -1).astype(np.float32)
                
                if _ML_MODEL_TYPE == 'json':
                    # Pure Python inference (ARM/mobile-friendly)
                    prediction_proba = _ML_MODEL.predict_proba(X_features)[0, 1]
                elif _ML_MODEL_TYPE == 'onnx':
                    # ONNX inference
                    input_name = _ML_SESSION.get_inputs()[0].name
                    output = _ML_SESSION.run(None, {input_name: X_features})
                    # Output is list of [labels, probabilities]
                    # probabilities is shape (1, 2) for binary classification
                    prediction_proba = float(output[1][0][1])  # Probability of class 1
                else:
                    # Standard scikit-learn inference
                    prediction_proba = _ML_MODEL.predict_proba(X_features)[0, 1]
            except Exception as e:
                logging.error("%s: ML prediction failed: %s", symbol, e)
                logging.exception(e)
                continue
            
            pos = state.get("positions", {}).get(symbol)
            
            logging.debug("%s: price=%.6f ML_prob=%.3f pos=%s",
                         symbol, current_price, prediction_proba, "OPEN" if pos else "NONE")
            
            # =======================
            # MANAGE EXISTING POSITION
            # =======================
            if pos is not None:
                entry_price = float(pos["entry_price"])
                amount = float(pos["amount"])
                buy_signal = pos.get("buy_signal", "")
                is_manual = (buy_signal == "MANUAL")
                
                # Update trailing stop using enhanced dynamic professional stop loss
                old_sl = pos.get("sl")
                old_highest = pos.get("highest_since_entry", entry_price)
                
                # Get candle low for reversal detection
                candle_low = float(df.iloc[-1]['low']) if len(df) > 0 else current_price
                
                # Check for early exit conditions FIRST (before updating SL)
                # This catches objectively bad trades before they hit full SL
                should_exit, exit_reason = check_early_exit_conditions(
                    symbol=symbol,
                    pos=pos,
                    current_price=current_price,
                    klines=klines,
                    df=df
                )
                
                if should_exit:
                    logging.warning("%s: EARLY EXIT triggered - %s (price=%.6f entry=%.6f)",
                                  symbol, exit_reason, current_price, entry_price)
                    reason = exit_reason
                else:
                    # Only update dynamic SL if early exit not triggered
                    # Call the enhanced professional stop loss function
                    update_dynamic_sl_pro(
                        symbol=symbol,
                        pos=pos,
                        current_price=current_price,
                        candle_high=candle_high,
                        candle_low=candle_low,
                        klines=klines,
                        state=state
                    )
                # Check if stop loss was updated
                if (pos.get("sl") != old_sl or pos.get("highest_since_entry") != old_highest):
                    state_modified = True
                    # Write state immediately when trailing stop is updated
                    if not safe_write_state(state):
                        logging.error("CRITICAL: Failed to save state after updating %s trailing stop", symbol)
                
                # STANDARD EXIT CONDITIONS (if not already triggered by early exit)
                if not should_exit:
                    if current_price <= pos.get("sl", float("-inf")):
                        should_exit = True
                        reason = "SL_HIT"
                    elif current_price >= pos.get("tp", float("inf")):
                        should_exit = True
                        reason = "TP_TARGET"
                
                # Execute exit if needed
                if should_exit:
                    if dry_run:
                        avg_exit_price = current_price
                        executed_qty = amount
                    else:
                        exit_order = safe_market_sell(api_key, api_secret, symbol, amount, state)
                        if not exit_order:
                            logging.error("%s: SELL order failed", symbol)
                            continue
                        executed_qty = float(exit_order.get("executedQty", "0") or 0)
                        executed_quote = float(exit_order.get("cummulativeQuoteQty", 0) or 0)
                        
                        if executed_qty <= 0:
                            logging.error("%s: SELL returned zero qty", symbol)
                            continue
                        
                        avg_exit_price = executed_quote / executed_qty if executed_qty > 0 else current_price
                    
                    pnl = (avg_exit_price - entry_price) * executed_qty
                    pnl_pct = ((avg_exit_price - entry_price) / entry_price) * 100
                    
                    msg = f"{symbol} SELL {executed_qty:.8f} @ {avg_exit_price:.6f} | {reason} | PnL ${pnl:.2f} ({pnl_pct:+.2f}%)"
                    logging.info(msg)
                    notify("Position Closed", msg)
                    
                    append_trade({
                        "time": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "side": "SELL",
                        "price": f"{avg_exit_price:.6f}",
                        "amount": f"{executed_qty:.8f}",
                        "notional_usd": f"{avg_exit_price * executed_qty:.2f}",
                        "pnl_usd": f"{pnl:.2f}",
                        "dry_run": str(dry_run),
                        "reason": reason,
                    })
                    
                    # Remove position and set cooldown
                    state.get("positions", {}).pop(symbol, None)
                    state.setdefault("cooldowns", {})[symbol] = datetime.now(timezone.utc).isoformat()
                    state_modified = True
                    
                    # Write state IMMEDIATELY after closing position to ensure persistence
                    if not safe_write_state(state):
                        logging.error("CRITICAL: Failed to save state after closing %s position", symbol)
                
                continue  # Skip to next symbol if we have a position
            
            # =======================
            # CHECK FOR NEW ENTRY
            # =======================
            
            # Cooldown check
            cooldowns = state.setdefault("cooldowns", {})
            cd_iso = cooldowns.get(symbol)
            in_cooldown = False
            if cd_iso:
                try:
                    cd_dt = datetime.fromisoformat(cd_iso)
                    if (datetime.now(timezone.utc) - cd_dt).total_seconds() < cooldown_seconds:
                        in_cooldown = True
                        logging.debug("%s: in cooldown", symbol)
                except Exception:
                    pass
            
            if in_cooldown:
                continue
            
            # ML ENTRY CONDITIONS
            entry_signal = False
            
            # Condition 1: Model confidence >= threshold (primary filter)
            if prediction_proba >= ml_prob_threshold:
                # Condition 2: Apply entry filter if specified
                if ml_entry_filter == "EMA200+":
                    # Check if price is above EMA 200
                    price_to_ema_200 = float(latest_row.get('price_to_ema_200', 0))
                    if price_to_ema_200 > 0:
                        entry_signal = True
                        logging.info("%s: ML ENTRY SIGNAL - Prob=%.3f EMA200_diff=%.2f%%",
                                   symbol, prediction_proba, price_to_ema_200)
                    else:
                        logging.debug("%s: ML signal rejected - price below EMA200 (diff=%.2f%%)",
                                     symbol, price_to_ema_200)
                elif ml_entry_filter == "None" or ml_entry_filter == "" or not ml_entry_filter:
                    # Optimized: threshold-only entry (no additional filters)
                    entry_signal = True
                    logging.info("%s: ML ENTRY SIGNAL - Prob=%.3f (threshold-only, optimized)",
                               symbol, prediction_proba)
                else:
                    logging.warning("%s: Unknown ML_ENTRY_FILTER '%s', using threshold-only",
                                  symbol, ml_entry_filter)
                    entry_signal = True
            else:
                logging.debug("%s: ML probability %.3f below threshold %.3f",
                            symbol, prediction_proba, ml_prob_threshold)
            
            if not entry_signal:
                continue
            
            # Calculate expected quantity
            expected_qty = position_usd / current_price
            
            # Check minimum quantity
            sf = SYMBOL_FILTERS.get(symbol.replace("/", ""))
            if sf:
                try:
                    min_qty = float(sf.get("minQty", 0))
                    min_notional = float(sf.get("minNotional", 0))
                    if expected_qty < min_qty or (expected_qty * current_price) < min_notional:
                        logging.warning("%s: expected qty %.8f below min, skipping", symbol, expected_qty)
                        continue
                except Exception:
                    pass
            
            # Execute BUY order
            if dry_run:
                simulated_slippage = 1.0 + float(get_cfg("SLIPPAGE_PCT", float, 0.001))
                avg_price = current_price * simulated_slippage
                executed_qty = expected_qty
                executed_quote = executed_qty * avg_price
                order_info = {
                    "status": "SIMULATED",
                    "executedQty": str(executed_qty),
                    "cummulativeQuoteQty": str(executed_quote)
                }
            else:
                order_info = safe_market_buy(api_key, api_secret, symbol, position_usd)
                if not order_info:
                    logging.error("%s: BUY order failed, skipping", symbol)
                    continue
                executed_qty = float(order_info.get("executedQty", "0") or 0)
                executed_quote = float(order_info.get("cummulativeQuoteQty", 0) or 0)
                
                if executed_qty <= 0:
                    logging.error("%s: BUY returned executedQty=0, skipping", symbol)
                    continue
                
                if executed_quote <= 0:
                    executed_quote = executed_qty * current_price
                    logging.warning("%s: cummulativeQuoteQty missing, estimated as %.2f", symbol, executed_quote)
                
                avg_price = executed_quote / executed_qty
            
            # Calculate stops using ML configuration
            fee_pct = float(get_cfg("FEE_PCT", float, 0.001))
            slip_pct = float(get_cfg("SLIPPAGE_PCT", float, 0.001))
            buffer_mult = 1.0 + fee_pct + slip_pct
            
            initial_sl = avg_price - (atr_val * ml_atr_mult * buffer_mult)
            initial_sl = max(initial_sl, avg_price * 0.98)  # 2% floor
            initial_tp = avg_price + (atr_val * ml_atr_mult * ml_rr_ratio * (1.0 - fee_pct - slip_pct))
            
            # Create position
            state.setdefault("positions", {})[symbol] = {
                "amount": executed_qty,
                "entry_price": avg_price,
                "sl": initial_sl,
                "initial_sl": initial_sl,
                "tp": initial_tp,
                "initial_tp": initial_tp,
                "locked_atr": float(atr_val),
                "risk": float(atr_val * ml_atr_mult),
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "buy_signal": "ML_STRATEGY",
                "highest_since_entry": avg_price,
                "last_ema_slope": 0.0,
                "ml_probability": float(prediction_proba),
            }
            
            state_modified = True
            
            # Write state IMMEDIATELY after opening position to ensure persistence
            if not safe_write_state(state):
                logging.error("CRITICAL: Failed to save state for %s position - position tracking may be lost", symbol)
            
            msg = f"{symbol} BUY {executed_qty:.8f} @ {avg_price:.6f} (${executed_quote:.2f}) | ML_Prob={prediction_proba:.3f} | SL={initial_sl:.6f} TP={initial_tp:.6f}"
            logging.info(msg)
            notify("ML Trade Placed", msg)
            
            append_trade({
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": "BUY",
                "price": f"{avg_price:.6f}",
                "amount": f"{executed_qty:.8f}",
                "notional_usd": f"{executed_quote:.2f}",
                "pnl_usd": "0.0",
                "dry_run": str(dry_run),
                "reason": "ML_STRATEGY",
                "sl": f"{initial_sl:.6f}",
                "tp": f"{initial_tp:.6f}",
                "ml_prob": f"{prediction_proba:.3f}",
            })
        
        except Exception as e:
            logging.exception("%s: Error in ML strategy cycle: %s", symbol, e)
            continue
    
    # Write state if modified
    if state_modified:
        if not safe_write_state(state):
            logging.error("CRITICAL: Failed to save state after ML strategy cycle")

# ---------- Main ----------
def main_loop():
    load_env_file(ENV_FILE)
    # Reinitialize logging after .env is loaded (in case PROJECT_DIR changed)
    _setup_logging()
    
    # Safety: prevent accidental live trading. If DRY_RUN is disabled, require explicit interactive confirmation
    dry_run_now = get_cfg("DRY_RUN", bool)
    if not dry_run_now:
        # If user already provided CONFIRM_LIVE=true in env, honor it (useful for non-interactive ops)
        if get_cfg("CONFIRM_LIVE", bool):
            logging.info("CONFIRM_LIVE=true detected – proceeding with live trading as requested.")
        else:
            # Non-interactive environment (no TTY) should not proceed
            if not sys.stdin or not sys.stdin.isatty():
                logging.error("DRY_RUN=false but no interactive terminal available and CONFIRM_LIVE not set. Refusing to start live.")
                notify("AutoTrader Safety", "Refusing to start: DRY_RUN=false but no interactive terminal and CONFIRM_LIVE not set.")
                return

            # Interactive confirmation: print a concise, non-secret summary and ask for an explicit confirmation phrase
            api_present = bool(CFG.get("API_KEY")) and bool(CFG.get("API_SECRET"))
            masked_key = None
            if api_present and CFG.get("API_KEY"):
                k = CFG.get("API_KEY")
                masked_key = f"{k[:4]}...{k[-4:]}"

            print("\n=== AUTO TRADER LIVE CONFIRMATION ===")
            print("This will start live trading. Review the configuration below and type the confirmation phrase to proceed.")
            print(f"API key present: {api_present} ({masked_key if masked_key else 'none'})")
            print(f"Symbols: {CFG.get('SYMBOLS')}")
            print(f"Position per trade (USD): {CFG.get('POSITION_USD')}")
            print(f"ATR_PERIOD: {CFG.get('ATR_PERIOD')}  ATR_MULT: {CFG.get('ATR_MULT')}  TAKE_RATIO: {CFG.get('TAKE_RATIO')}")
            print(f"EMA_FAST/EMA_SLOW: {CFG.get('EMA_FAST')}/{CFG.get('EMA_SLOW')}  EMA_TREND_PERCENT: {CFG.get('EMA_TREND_PERCENT')}")
            print(f"COOLDOWN_SECONDS: {CFG.get('COOLDOWN_SECONDS')}  NOTIFY: {CFG.get('NOTIFY')}\n")

            print("To confirm live trading, type exactly: CONFIRM_LIVE and press Enter.")
            try:
                resp = input("Confirmation: ").strip()
            except Exception:
                resp = ""

            if resp == "CONFIRM_LIVE":
                CFG["CONFIRM_LIVE"] = "true"
                logging.info("User confirmed live trading via interactive prompt.")
            else:
                logging.error("Interactive confirmation failed or cancelled – not starting live trading.")
                notify("AutoTrader Safety", "Live start cancelled by user (interactive confirmation failed).")
                return
    
    state = read_state()
    state = initialize_state(CFG, state)
    write_state(state)
    notify("AutoTrader V6", f"Started (DRY_RUN={CFG.get('DRY_RUN')})")
    logging.info("AutoTrader V6 starting config: %s", {k: CFG[k] for k in ("SYMBOLS","POSITION_USD","FAST_SMA","SLOW_SMA","TIMEFRAME","POLL_SECONDS","DRY_RUN")})
    
    # Parse symbols from config
    symbols = [s.strip().upper() for s in CFG.get("SYMBOLS", "").split(",") if s.strip()]
    
    if not symbols:
        logging.error("No symbols configured in SYMBOLS setting. Exiting.")
        notify("AutoTrader Error", "No symbols configured")
        return
    
    logging.info("Trading symbols: %s", ", ".join(symbols))
    
    # Select strategy: Use RUN_CYCLE_STRATEGY env var (default: "cycle2")
    state = read_state()
    state = initialize_state(CFG, state)
    write_state(state)
    notify("AutoTrader V8", f"Started (DRY_RUN={CFG.get('DRY_RUN')})")
    
    symbols = [s.strip().upper() for s in CFG.get("SYMBOLS", "").split(",") if s.strip()]
    
    if not symbols:
        logging.error("No symbols configured. Exiting.")
        notify("AutoTrader Error", "No symbols configured")
        return
    
    logging.info("Trading symbols: %s", ", ".join(symbols))
    
    # ===== MODIFIED STRATEGY SELECTION =====
    strategy = get_cfg("RUN_CYCLE_STRATEGY", str, "rsi_momentum").lower()
    
    if strategy in ("cycle4", "4", "ml", "machine_learning"):
        logging.info("Using RUN_CYCLE4 strategy (ML-powered RandomForest with 92 features)")
        use_ml_strategy = True
        use_rsi_momentum = False
        use_cycle2 = False
        use_cycle1 = False
    elif strategy in ("cycle3", "rsi", "momentum", "bollinger"):
        logging.info("Using RSI MOMENTUM + Bollinger Squeeze strategy")
        use_ml_strategy = False
        use_rsi_momentum = True
        use_cycle2 = False
        use_cycle1 = False
    elif strategy in ("cycle2", "2", "pullback"):
        logging.info("Using RUN_CYCLE2 strategy (HTF EMA21 + 15m pullback)")
        use_ml_strategy = False
        use_rsi_momentum = False
        use_cycle2 = True
        use_cycle1 = False
    elif strategy in ("cycle1", "1", "ema"):
        logging.info("Using RUN_CYCLE strategy (multi-indicator EMA-based)")
        use_ml_strategy = False
        use_rsi_momentum = False
        use_cycle2 = False
        use_cycle1 = True
    else:
        logging.error("Unknown RUN_CYCLE_STRATEGY=%s, defaulting to rsi_momentum", strategy)
        use_ml_strategy = True
        use_rsi_momentum = False
        use_cycle2 = False
        use_cycle1 = False
    
    try:
        while True:
            try:
                if use_ml_strategy:
                    # ML Strategy (cycle4) - RandomForest with 92 features
                    try:
                        run_cycle4_ml_strategy(CFG)
                    except Exception as e:
                        logging.exception("Error in run_cycle4_ml_strategy: %s", e)
                        notify("AutoTrader Error", f"ml_strategy error: {e}")
                        time.sleep(2)
                elif use_rsi_momentum:
                    # RSI Momentum strategy (recommended)
                    try:
                        run_cycle_rsi_momentum(CFG)
                    except Exception as e:
                        logging.exception("Error in run_cycle_rsi_momentum: %s", e)
                        notify("AutoTrader Error", f"rsi_momentum error: {e}")
                        time.sleep(2)
                elif use_cycle2:
                    # Cycle2: per-symbol execution
                    for symbol in symbols:
                        try:
                            run_cycle2(symbol)
                        except Exception as e:
                            logging.exception("Error in run_cycle2 for %s: %s", symbol, e)
                            time.sleep(2)
                elif use_cycle1:
                    # Cycle1: all-symbols in one call
                    try:
                        run_cycle(CFG)
                    except Exception as e:
                        logging.exception("Error in run_cycle: %s", e)
                        time.sleep(2)
            except Exception as e:
                logging.exception("Unexpected error in main cycle: %s", e)
                time.sleep(5)
            
            time.sleep(max(5, int(get_cfg("POLL_SECONDS", int, 30))))
    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting.")
        notify("AutoTrader V8", "Stopped by user")
if __name__ == "__main__":
    main_loop()
