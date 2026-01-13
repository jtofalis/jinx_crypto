#!/usr/bin/env python3
"""
Real-Time Trading Bot Dashboard
Lightweight Flask app for monitoring bot performance via web browser
Access from phone: http://RASPBERRY_PI_IP:5000
"""

from flask import Flask, render_template, jsonify
import json
import os
import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import threading
import time
import urllib.request
import urllib.parse

app = Flask(__name__)

# Configuration
# Point to parent directory where bot runs and creates files
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(PROJECT_DIR, "autotrader_state.json")
TRADES_CSV = os.path.join(PROJECT_DIR, "autotrader_trades.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "autotrader.log")

# Cache for performance metrics (updated every 30 seconds)
_metrics_cache = {
    "last_update": 0,
    "data": None
}

def read_state():
    """Read current bot state"""
    try:
        if not os.path.exists(STATE_FILE):
            return {}
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading state: {e}")
        return {}

def read_trades():
    """Read all trades from CSV"""
    trades = []
    try:
        if not os.path.exists(TRADES_CSV):
            return trades
        with open(TRADES_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
    except Exception as e:
        print(f"Error reading trades: {e}")
    return trades

def read_recent_logs(lines=50):
    """Read last N lines from log file"""
    try:
        if not os.path.exists(LOG_FILE):
            return []
        with open(LOG_FILE, 'r') as f:
            return f.readlines()[-lines:]
    except Exception:
        return []

def fetch_live_price(symbol):
    """Fetch current price from Binance API"""
    try:
        symbol_formatted = symbol.replace("/", "")
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol_formatted}"
        req = urllib.request.Request(url, headers={"User-Agent": "Dashboard/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return float(data.get("price", 0))
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def calculate_metrics():
    """Calculate comprehensive performance metrics"""
    now = time.time()
    
    # Use cache if recent (30 second freshness)
    if now - _metrics_cache["last_update"] < 30 and _metrics_cache["data"]:
        return _metrics_cache["data"]
    
    state = read_state()
    trades = read_trades()
    
    # --- POSITIONS ---
    positions = state.get("positions", {})
    open_positions = []
    total_unrealized_pnl = 0.0
    
    for symbol, pos in positions.items():
        try:
            entry_price = float(pos.get("entry_price", 0))
            amount = float(pos.get("amount", 0))
            
            # Fetch live price from Binance
            current_price = fetch_live_price(symbol)
            if current_price is None or current_price <= 0:
                current_price = entry_price  # Fallback to entry price if fetch fails
            
            pnl = (current_price - entry_price) * amount
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0
            
            open_positions.append({
                "symbol": symbol,
                "amount": amount,
                "entry_price": entry_price,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "sl": float(pos.get("sl", 0)),
                "tp": float(pos.get("tp", 0)),
                "opened_at": pos.get("opened_at", ""),
                "buy_signal": pos.get("buy_signal", "")
            })
            total_unrealized_pnl += pnl
        except Exception as e:
            print(f"Error processing position {symbol}: {e}")
    
    # --- TRADE STATISTICS ---
    total_trades = len(trades)
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0.0
    total_volume = 0.0
    
    trades_by_day = defaultdict(lambda: {"count": 0, "pnl": 0.0, "volume": 0.0})
    trades_by_symbol = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0})
    
    for trade in trades:
        try:
            if trade.get("side") != "SELL":
                continue  # Only count completed trades
            
            pnl = float(trade.get("pnl_usd", 0))
            notional = float(trade.get("notional_usd", 0))
            symbol = trade.get("symbol", "UNKNOWN")
            trade_time = trade.get("time", "")
            
            total_pnl += pnl
            total_volume += notional
            
            if pnl > 0:
                winning_trades += 1
                trades_by_symbol[symbol]["wins"] += 1
            elif pnl < 0:
                losing_trades += 1
                trades_by_symbol[symbol]["losses"] += 1
            
            # Group by day
            try:
                dt = datetime.fromisoformat(trade_time)
                day_key = dt.strftime("%Y-%m-%d")
                trades_by_day[day_key]["count"] += 1
                trades_by_day[day_key]["pnl"] += pnl
                trades_by_day[day_key]["volume"] += notional
            except Exception:
                pass
            
            # Group by symbol
            trades_by_symbol[symbol]["count"] += 1
            trades_by_symbol[symbol]["pnl"] += pnl
        except Exception as e:
            print(f"Error processing trade: {e}")
    
    # Win rate
    completed_trades = winning_trades + losing_trades
    win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0
    
    # Profit factor
    gross_profit = sum(float(t.get("pnl_usd", 0)) for t in trades 
                       if t.get("side") == "SELL" and float(t.get("pnl_usd", 0)) > 0)
    gross_loss = abs(sum(float(t.get("pnl_usd", 0)) for t in trades 
                         if t.get("side") == "SELL" and float(t.get("pnl_usd", 0)) < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
    # Recent performance (last 24h)
    cutoff_24h = datetime.now(timezone.utc) - timedelta(hours=24)
    recent_trades = []
    recent_pnl_24h = 0.0
    recent_volume_24h = 0.0
    
    for trade in reversed(trades[-50:]):  # Last 50 trades
        try:
            trade_time = datetime.fromisoformat(trade.get("time", ""))
            if trade_time.replace(tzinfo=timezone.utc) >= cutoff_24h:
                recent_trades.append(trade)
                if trade.get("side") == "SELL":
                    recent_pnl_24h += float(trade.get("pnl_usd", 0))
                    recent_volume_24h += float(trade.get("notional_usd", 0))
        except Exception:
            pass
    
    # Cooldowns
    cooldowns = state.get("cooldowns", {})
    active_cooldowns = []
    for symbol, cd_time in cooldowns.items():
        try:
            cd_dt = datetime.fromisoformat(cd_time)
            time_left = 300 - (datetime.now(timezone.utc) - cd_dt).total_seconds()
            if time_left > 0:
                active_cooldowns.append({
                    "symbol": symbol,
                    "time_left": int(time_left)
                })
        except Exception:
            pass
    
    # Compile metrics
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_pnl": round(total_pnl, 2),
            "unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_pnl_realized_unrealized": round(total_pnl + total_unrealized_pnl, 2),
            "total_trades": completed_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "total_volume": round(total_volume, 2),
            "open_positions_count": len(open_positions),
        },
        "recent_24h": {
            "trades": len(recent_trades),
            "pnl": round(recent_pnl_24h, 2),
            "volume": round(recent_volume_24h, 2),
        },
        "positions": open_positions,
        "cooldowns": active_cooldowns,
        "trades_by_symbol": [
            {
                "symbol": symbol,
                "trades": data["count"],
                "pnl": round(data["pnl"], 2),
                "wins": data["wins"],
                "losses": data["losses"],
                "wr": round(data["wins"] / (data["wins"] + data["losses"]) * 100, 1) if (data["wins"] + data["losses"]) > 0 else 0
            }
            for symbol, data in sorted(trades_by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True)
        ],
        "trades_by_day": [
            {
                "date": day,
                "trades": data["count"],
                "pnl": round(data["pnl"], 2),
                "volume": round(data["volume"], 2),
            }
            for day, data in sorted(trades_by_day.items(), reverse=True)[:30]  # Last 30 days
        ],
        "recent_trades": trades[-20:][::-1],  # Last 20 trades
    }
    
    # Update cache
    _metrics_cache["last_update"] = now
    _metrics_cache["data"] = metrics
    
    return metrics

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics (JSON)"""
    metrics = calculate_metrics()
    return jsonify(metrics)

@app.route('/api/logs')
def api_logs():
    """API endpoint for recent logs"""
    logs = read_recent_logs(100)
    return jsonify({"logs": logs})

@app.route('/api/state')
def api_state():
    """API endpoint for raw state (debugging)"""
    state = read_state()
    return jsonify(state)

# ============================================================================
# BACKGROUND UPDATER (refresh metrics cache)
# ============================================================================

def background_updater():
    """Background thread to refresh metrics cache"""
    while True:
        try:
            calculate_metrics()  # This updates the cache
            time.sleep(30)  # Update every 30 seconds
        except Exception as e:
            print(f"Background updater error: {e}")
            time.sleep(60)

# Start background thread
updater_thread = threading.Thread(target=background_updater, daemon=True)
updater_thread.start()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Trading Bot Dashboard Starting...")
    print("=" * 60)
    print(f"Project Dir: {PROJECT_DIR}")
    print(f"State File: {STATE_FILE}")
    print(f"Trades CSV: {TRADES_CSV}")
    print("")
    print("Access dashboard at:")
    print("  Local:    http://localhost:5000")
    print("  Network:  http://YOUR_PI_IP:5000")
    print("")
    print("From your phone:")
    print("  1. Connect to same WiFi as Raspberry Pi")
    print("  2. Open browser to http://RASPBERRY_PI_IP:5000")
    print("=" * 60)
    
    # Run Flask app (accessible from network)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
