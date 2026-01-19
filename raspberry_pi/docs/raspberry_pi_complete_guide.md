# 🍓 COMPLETE RASPBERRY PI MIGRATION GUIDE

## Overview: What You're About To Do

You're moving your crypto trading bot from **Termux on Android** → **Raspberry Pi 4B (4GB)**

**Why this is awesome:**
- 24/7 stable operation (no phone battery/heat issues)
- Full Linux environment (easier debugging)
- SSH access from anywhere
- Web dashboard for monitoring
- Better performance & reliability

---

## Phase 1: Initial Raspberry Pi Setup (30-60 min)

### 1.1 Flash Raspberry Pi OS

**On your PC:**
1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Insert SD card (minimum 16GB recommended)
3. Flash "Raspberry Pi OS Lite (64-bit)" or full desktop version
4. **IMPORTANT**: Before flashing, click ⚙️ (settings):
   - Enable SSH
   - Set username: `pi`
   - Set password: (your choice)
   - Configure WiFi (optional, or use ethernet)

### 1.2 First Boot

1. Insert SD card into Raspberry Pi
2. Connect power + ethernet (or use WiFi from step above)
3. Wait 2-3 minutes for first boot

### 1.3 Find Your Pi's IP Address

**Option A - From your router:**
- Check connected devices, look for "raspberrypi"

**Option B - IP scanner app on phone:**
- Download "Fing" app
- Scan network for Raspberry Pi

**Option C - Connect monitor/keyboard:**
- Login and run: `hostname -I`

### 1.4 SSH Into Your Pi

**From your phone (Termux):**
```bash
ssh pi@192.168.1.XXX
# Replace XXX with your Pi's IP
# Enter password you set during flash
```

**From Windows PC:**
```bash
ssh pi@192.168.1.XXX
```

You're now on your Raspberry Pi! 🎉

---

## Phase 2: System Preparation (15 min)

### 2.1 Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Install Required Packages

```bash
# Python and essential tools
sudo apt install -y python3 python3-pip python3-dev python3-numpy git screen htop

# Optional but recommended
sudo apt install -y vim nano curl wget
```

### 2.3 Verify Python

```bash
python3 --version
# Should show Python 3.9+ (Pi OS default)
```

---

## Phase 3: Transfer Your Trading Bot (15 min)

### 3.1 Create Project Directory

```bash
cd ~
mkdir trading_bot
cd trading_bot
```

### 3.2 Transfer Files from Your Computer

**Option A - Using SCP (from Windows/Linux PC):**
```bash
# On your PC, from the folder containing ML_OPTIMIZED_DEPLOYMENT
scp -r ML_OPTIMIZED_DEPLOYMENT pi@192.168.1.XXX:~/trading_bot/
```

**Option B - Using FileZilla/WinSCP (GUI):**
1. Download [WinSCP](https://winscp.net/)
2. Connect to: `pi@192.168.1.XXX`
3. Drag & drop `ML_OPTIMIZED_DEPLOYMENT` folder

**Option C - Git (if you have a repo):**
```bash
git clone <your-repo-url>
cd ML_OPTIMIZED_DEPLOYMENT
```

### 3.3 Verify Files Transferred

```bash
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT
ls -la
```

You should see:
```
autotrader_ml_optimized.py
dashboard.py
feature_engineering.py
config.py
.env
models/
templates/
etc...
```

---

## Phase 4: Install Python Dependencies (10 min)

### 4.1 Install Bot Requirements

```bash
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT
pip3 install -r requirements.txt
```

**Common issues:**
- `ta-lib` might fail → skip it (optional dependency)
- `matplotlib` takes time → be patient
- If memory issues → add swap space (see troubleshooting)

### 4.2 Fix Startup Script Paths

The `start_trader.sh` has Termux-specific paths. Fix it:

```bash
nano start_trader.sh
```

**Change line 1 from:**
```bash
#!/data/data/com.termux/files/usr/bin/bash
```

**To:**
```bash
#!/bin/bash
```

Save (Ctrl+O, Enter, Ctrl+X)

Make executable:
```bash
chmod +x start_trader.sh
chmod +x start_pi.sh
```

---

## Phase 5: Configure Your Bot (5 min)

### 5.1 Edit .env File

```bash
nano .env
```

Add your API keys:
```bash
API_KEY=your_binance_api_key_here
API_SECRET=your_binance_secret_here

# Optional: Start in dry-run mode for testing
DRY_RUN=true
```

Save (Ctrl+O, Enter, Ctrl+X)

### 5.2 Verify Configuration

```bash
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', os.getenv('API_KEY')[:10] + '...')"
```

Should print first 10 chars of your API key.

---

## Phase 6: Test Run (5 min)

### 6.1 Start Bot in Foreground

```bash
python3 autotrader_ml_optimized.py
```

Watch for:
- ✅ "ML model loaded"
- ✅ "Symbol filters loaded"
- ✅ Position updates

If errors appear, check logs:
```bash
tail -f autotrader.log
```

**Stop bot:** Press Ctrl+C

---

## Phase 7: Set Up Web Dashboard (10 min)

### 7.1 Test Dashboard

```bash
python3 dashboard.py
```

You should see:
```
Trading Bot Dashboard Starting...
Access dashboard at:
  Local:    http://localhost:5000
  Network:  http://192.168.1.XXX:5000
```

### 7.2 Access from Your Phone

1. Connect phone to same WiFi
2. Open browser (Chrome/Safari)
3. Go to: `http://192.168.1.XXX:5000`
   (Replace XXX with your Pi's actual IP)
4. Bookmark it!

You should see your trading dashboard with live stats 🎉

**Stop dashboard:** Press Ctrl+C

---

## Phase 8: Run in Background (Permanent Setup)

### 8.1 Using Screen (Simple Method)

**Start bot:**
```bash
screen -S trader
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT
python3 autotrader_ml_optimized.py
# Press Ctrl+A then D to detach
```

**Start dashboard:**
```bash
screen -S dashboard
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT
python3 dashboard.py
# Press Ctrl+A then D to detach
```

**View running processes:**
```bash
screen -ls
```

**Reattach to view:**
```bash
screen -r trader    # View bot
screen -r dashboard # View dashboard
```

**Quick launcher (uses menu):**
```bash
./start_pi.sh
# Choose option 4 - "Start both in background"
```

### 8.2 Using Systemd (Advanced - Auto-start on boot)

**Create bot service:**
```bash
sudo nano /etc/systemd/system/cryptobot.service
```

Paste:
```ini
[Unit]
Description=ML Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/trading_bot/ML_OPTIMIZED_DEPLOYMENT
ExecStart=/usr/bin/python3 autotrader_ml_optimized.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Create dashboard service:**
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
WorkingDirectory=/home/pi/trading_bot/ML_OPTIMIZED_DEPLOYMENT
ExecStart=/usr/bin/python3 dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl enable cryptobot dashboard
sudo systemctl start cryptobot dashboard
```

**Check status:**
```bash
sudo systemctl status cryptobot
sudo systemctl status dashboard
```

**View logs:**
```bash
sudo journalctl -u cryptobot -f    # Bot logs
sudo journalctl -u dashboard -f    # Dashboard logs
```

**Restart services:**
```bash
sudo systemctl restart cryptobot
sudo systemctl restart dashboard
```

Now your bot starts automatically on boot! 🚀

---

## Phase 9: SSH from Your Phone Anytime

### 9.1 Install SSH App on Phone

**Android:**
- [Termux](https://f-droid.org/en/packages/com.termux/) (you have this)
- [JuiceSSH](https://play.google.com/store/apps/details?id=com.sonelli.juicessh) (recommended)

**iOS:**
- [Terminus](https://apps.apple.com/us/app/termius-ssh-client/id549039908)
- [Prompt](https://apps.apple.com/us/app/prompt-3/id1594420480)

### 9.2 Set Up SSH Keys (No Password Needed)

**On your phone (Termux):**
```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519
# Press Enter 3 times (no passphrase)

# Copy key to Pi
ssh-copy-id pi@192.168.1.XXX
# Enter password one last time
```

Now you can SSH without password:
```bash
ssh pi@192.168.1.XXX
```

### 9.3 Useful SSH Commands

```bash
# Quick check bot status
screen -r trader

# View logs
tail -50 autotrader.log

# Check system resources
htop

# Restart bot
sudo systemctl restart cryptobot

# View recent trades
tail -20 autotrader_trades.csv | column -s, -t
```

### 9.4 Remote Access from Anywhere (VPN)

**Install Tailscale (easiest VPN):**
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

- Install Tailscale app on phone
- Now you can SSH even outside your home WiFi
- Access dashboard via Tailscale IP

---

## Phase 10: Monitoring & Maintenance

### 10.1 Daily Checks

**From phone browser:**
- Open dashboard: `http://192.168.1.XXX:5000`
- Check P&L, open positions, recent trades

**From SSH (if needed):**
```bash
screen -r trader  # View bot
tail autotrader.log
```

### 10.2 Weekly Tasks

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Update Python packages
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT
pip3 install --upgrade -r requirements.txt

# Restart services
sudo systemctl restart cryptobot dashboard
```

### 10.3 Backup Important Files

```bash
# From your PC, backup config and data
scp pi@192.168.1.XXX:~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/.env ./backup/
scp pi@192.168.1.XXX:~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader_trades.csv ./backup/
scp pi@192.168.1.XXX:~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader_state.json ./backup/
```

---

## Troubleshooting

### Bot won't start
```bash
# Check Python errors
python3 autotrader_ml_optimized.py

# Check dependencies
pip3 list | grep -i pandas

# View system logs
sudo journalctl -u cryptobot -n 50
```

### Can't access dashboard from phone
```bash
# 1. Verify dashboard is running
ps aux | grep dashboard

# 2. Test from Pi itself
curl http://localhost:5000

# 3. Check IP address
hostname -I

# 4. Verify same WiFi network
```

### Out of memory errors
```bash
# Add swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### SD card corruption (power loss)
- Always use `sudo shutdown -h now` before unplugging
- Consider getting UPS (power backup) for Pi
- Keep backups of .env and trades CSV

---

## Performance Comparison

| Metric | Phone (Termux) | Raspberry Pi 4B |
|--------|---------------|-----------------|
| Stability | ★★☆☆☆ | ★★★★★ |
| Power usage | Variable | ~3W constant |
| Heat | Issues | Cool/heatsink |
| SSH access | Complex | Easy |
| Monitoring | Manual | Web dashboard |
| Cost per month | Free (but wears battery) | ~$0.50 electricity |
| Reliability | 60-70% | 99%+ |

---

## Next Steps & Ideas

1. ✅ **Add email alerts**: Notify on trades via Gmail API
2. ✅ **Set up Grafana**: Professional metrics dashboard
3. ✅ **Add Telegram bot**: Control bot via chat commands
4. ✅ **Database logging**: Store trades in SQLite/PostgreSQL
5. ✅ **Multi-exchange**: Add more exchanges beyond Binance
6. ✅ **Backtesting UI**: Web interface for strategy testing
7. ✅ **Auto-rebalance**: Portfolio management features

---

## Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Raspberry Pi 4B (4GB) | $55 | One-time |
| Power supply (official) | $8 | One-time |
| MicroSD card (32GB) | $10 | One-time |
| Case with fan/heatsink | $12 | Optional but recommended |
| **Total** | **~$85** | Pays for itself in uptime |
| Monthly electricity | $0.50 | 3W × 24h × 30d |

---

## Final Checklist

- [ ] Raspberry Pi flashed with OS
- [ ] SSH enabled and working
- [ ] Files transferred to Pi
- [ ] Python dependencies installed
- [ ] `.env` configured with API keys
- [ ] Bot tested in foreground
- [ ] Dashboard accessible from phone
- [ ] Services running in background (screen or systemd)
- [ ] SSH access from phone working
- [ ] Dashboard bookmarked on phone
- [ ] Backup strategy in place

---

## Support & Resources

- **Your logs**: `~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader.log`
- **Raspberry Pi docs**: https://www.raspberrypi.com/documentation/
- **Python on Pi**: Works exactly like any Linux system
- **Screen tutorial**: `man screen` or `screen -h`

---

**You're all set!** Your bot is now running 24/7 on enterprise-grade hardware (sort of 😄) with professional monitoring. Welcome to the world of automated trading on the Raspberry Pi! 🍓🤖💰

If you run into issues, check the logs first, then the troubleshooting section above.
