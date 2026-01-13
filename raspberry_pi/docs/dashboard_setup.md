# 🎯 RASPBERRY PI DASHBOARD SETUP GUIDE

## What You Just Created

A **real-time web dashboard** that lets you monitor your trading bot from any device (phone, tablet, laptop) on the same network. No need to SSH in just to check performance!

## Features

✅ **Real-time metrics**: P&L, win rate, profit factor, positions  
✅ **Live position tracking**: See all open trades with unrealized P&L  
✅ **Trade history**: Last 20 trades with entry/exit details  
✅ **Performance by coin**: Which symbols are making you money  
✅ **Live logs**: Last 100 log lines, auto-refreshing  
✅ **Mobile-optimized**: Works perfectly on phone browsers  
✅ **Auto-refresh**: Updates every 10 seconds automatically

---

## Installation on Raspberry Pi

### 1. Install Flask (one-time setup)
```bash
pip3 install flask
```

### 2. Transfer files to Pi
Make sure these files are in your `ML_OPTIMIZED_DEPLOYMENT` folder:
```
ML_OPTIMIZED_DEPLOYMENT/
├── dashboard.py              # ← Dashboard server (NEW)
├── templates/
│   └── dashboard.html        # ← Web interface (NEW)
├── autotrader_state.json     # Bot state (auto-created)
├── autotrader_trades.csv     # Trade log (auto-created)
└── autotrader.log            # Log file (auto-created)
```

### 3. Start the dashboard
```bash
cd ML_OPTIMIZED_DEPLOYMENT
python3 dashboard.py
```

You'll see:
```
Trading Bot Dashboard Starting...
Access dashboard at:
  Local:    http://localhost:5000
  Network:  http://YOUR_PI_IP:5000
```

### 4. Find your Raspberry Pi's IP address
On the Pi:
```bash
hostname -I
```
Example output: `192.168.1.100`

### 5. Access from your phone
- Connect phone to same WiFi as Raspberry Pi
- Open browser (Chrome/Safari)
- Go to: `http://192.168.1.100:5000`
- Bookmark it!

---

## Running Both Bot + Dashboard Together

You have two options:

### Option A: Two Terminal Windows (Easiest for testing)

**Terminal 1** - Trading bot:
```bash
cd ML_OPTIMIZED_DEPLOYMENT
./start_trader.sh
```

**Terminal 2** - Dashboard:
```bash
cd ML_OPTIMIZED_DEPLOYMENT
python3 dashboard.py
```

### Option B: Run Dashboard in Background (Recommended for 24/7)

Using `screen` (keeps running when you disconnect):

**Start the bot**:
```bash
screen -S trader
cd ML_OPTIMIZED_DEPLOYMENT
./start_trader.sh
# Press Ctrl+A, then D to detach
```

**Start the dashboard**:
```bash
screen -S dashboard
cd ML_OPTIMIZED_DEPLOYMENT
python3 dashboard.py
# Press Ctrl+A, then D to detach
```

**Reattach anytime**:
```bash
screen -r trader    # View bot
screen -r dashboard # View dashboard
```

**List running screens**:
```bash
screen -ls
```

---

## Auto-Start on Boot (Optional)

Create systemd service for dashboard:

```bash
sudo nano /etc/systemd/system/dashboard.service
```

Paste:
```ini
[Unit]
Description=Trading Bot Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ML_OPTIMIZED_DEPLOYMENT
ExecStart=/usr/bin/python3 dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable dashboard
sudo systemctl start dashboard
sudo systemctl status dashboard
```

---

## Accessing from Outside Your Home Network (Advanced)

If you want to check your bot while away from home:

### Option 1: Tailscale VPN (Easiest & Secure)
1. Install on Pi: `curl -fsSL https://tailscale.com/install.sh | sh`
2. Install Tailscale app on phone
3. Access dashboard via Tailscale IP (always works, secure)

### Option 2: Port Forwarding (Less secure)
1. Router settings → Port forwarding
2. Forward external port 5000 → Pi's IP:5000
3. Access via: `http://YOUR_PUBLIC_IP:5000`
4. **Not recommended** without HTTPS/authentication

---

## Dashboard Screenshots (What You'll See)

```
┌─────────────────────────────────────────────┐
│      🤖 Trading Bot Dashboard               │
│      Last updated: 2026-01-03 10:45:23     │
│      [🔄 Refresh Now]                       │
└─────────────────────────────────────────────┘

┌──────────────┬──────────────┬──────────────┐
│ Total P&L    │ Win Rate     │ Profit Factor│
│ $2,468.50    │ 53.3%        │ 2.25         │
│ +$124 unreal │ 16W / 14L    │ Risk:Reward  │
└──────────────┴──────────────┴──────────────┘

┌──────────────┬──────────────┬──────────────┐
│ Open Pos.    │ 24h Perf.    │ Total Trades │
│ 3            │ +$45.20      │ 87           │
│ $1,234 vol   │ 5 trades     │ Completed    │
└──────────────┴──────────────┴──────────────┘

📊 Open Positions
┌────────┬────────┬─────────┬────────┬─────────┐
│ Symbol │ Entry  │ Current │ Amount │ P&L     │
├────────┼────────┼─────────┼────────┼─────────┤
│ BTC/USDT│ $42,150│ $42,380│ 0.0123 │ +$2.83 │
│ ETH/USDT│ $2,240 │ $2,265 │ 0.4500 │ +$11.25│
│ SOL/USDT│ $98.50 │ $97.20 │ 5.2000 │ -$6.76 │
└────────┴────────┴─────────┴────────┴─────────┘

📈 Recent Trades (Last 20)
[Live updating table with your latest trades]

💰 Performance by Symbol
[Which coins are profitable]

📝 Recent Logs
[Live scrolling log output]
```

---

## Troubleshooting

### Dashboard won't start
```bash
# Check if Flask is installed
pip3 list | grep Flask

# Install if missing
pip3 install flask
```

### Can't access from phone
```bash
# 1. Verify Pi's IP
hostname -I

# 2. Check if dashboard is running
ps aux | grep dashboard.py

# 3. Test from Pi itself
curl http://localhost:5000

# 4. Check firewall (usually not enabled on Pi)
sudo ufw status
```

### Dashboard shows no data
- Make sure the bot has created `autotrader_trades.csv`
- Check that `autotrader_state.json` exists
- Run the bot for a few minutes first

---

## Next Steps

1. ✅ **Test locally**: Run dashboard, access from Pi's browser
2. ✅ **Test from phone**: Access via Pi's IP on phone
3. ✅ **Bookmark on phone**: Add to home screen for quick access
4. ✅ **Set up auto-start**: Use systemd service
5. ✅ **Optional VPN**: Install Tailscale for remote access

---

## Why This is Better Than SSH

| Feature | SSH from Phone | Web Dashboard |
|---------|---------------|---------------|
| Access | Terminal only | Any browser |
| Visuals | Text only | Charts, colors, tables |
| Real-time | Manual refresh | Auto-refresh |
| Multi-device | One at a time | Multiple viewers |
| Easy to use | Need SSH app | Just a URL |
| Quick check | 30+ seconds | 5 seconds |

---

## Customization Ideas

Want to add more features? Edit `dashboard.py`:

- **Email alerts**: When P&L hits threshold
- **Charts**: Plot equity curve over time
- **Mobile push notifications**: Via Pushover/Telegram
- **Password protection**: Add Flask-Login
- **HTTPS**: Use Let's Encrypt + nginx
- **Dark mode**: CSS theme toggle

---

**You now have a professional monitoring solution!** 🎉

Questions? Check the bot logs or state file for troubleshooting.
