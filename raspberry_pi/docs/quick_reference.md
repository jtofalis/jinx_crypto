# 📋 RASPBERRY PI QUICK REFERENCE CARD

## Essential Commands (Copy/Paste Ready)

### 🚀 Starting Services

```bash
# Quick launcher (interactive menu)
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT
./start_pi.sh

# Manual start in background
screen -dmS trader bash -c "cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT && python3 autotrader_ml_optimized.py"
screen -dmS dashboard bash -c "cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT && python3 dashboard.py"
```

### 👀 Viewing Running Processes

```bash
# List all screen sessions
screen -ls

# Attach to bot
screen -r trader

# Attach to dashboard
screen -r dashboard

# Detach from screen
Ctrl+A, then D
```

### 🛑 Stopping Services

```bash
# Kill specific screen
screen -S trader -X quit
screen -S dashboard -X quit

# Kill both
screen -S trader -X quit && screen -S dashboard -X quit

# If using systemd
sudo systemctl stop cryptobot dashboard
```

### 📊 Checking Status

```bash
# Check if bot is running
ps aux | grep autotrader

# Check if dashboard is running  
ps aux | grep dashboard

# View last 50 log lines
tail -50 ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader.log

# Follow logs in real-time
tail -f ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader.log

# View recent trades
tail -20 ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader_trades.csv | column -s, -t

# Check system resources
htop
```

### 🌐 Network & Access

```bash
# Find Pi's IP address
hostname -I

# Dashboard URL (from same network)
http://$(hostname -I | awk '{print $1}'):5000

# Test dashboard locally
curl http://localhost:5000
```

### 🔧 Systemd Service Commands

```bash
# Start services
sudo systemctl start cryptobot dashboard

# Stop services
sudo systemctl stop cryptobot dashboard

# Restart services
sudo systemctl restart cryptobot dashboard

# Check status
sudo systemctl status cryptobot
sudo systemctl status dashboard

# View logs
sudo journalctl -u cryptobot -f
sudo journalctl -u dashboard -f

# Enable auto-start on boot
sudo systemctl enable cryptobot dashboard

# Disable auto-start
sudo systemctl disable cryptobot dashboard
```

### 📝 File Locations

```bash
# Project directory
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT

# Configuration
nano ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/.env

# Bot script
nano ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader_ml_optimized.py

# Dashboard
nano ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/dashboard.py

# Logs
tail ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader.log

# Trades history
tail ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader_trades.csv

# State file (positions)
cat ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader_state.json | jq .
```

### 🔄 Updates & Maintenance

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT
pip3 install --upgrade -r requirements.txt

# Restart services after update
sudo systemctl restart cryptobot dashboard
# OR
screen -S trader -X quit && screen -S dashboard -X quit
./start_pi.sh  # Choose option 4
```

### 💾 Backup

```bash
# Quick backup (from Pi)
cd ~
tar -czf trading_bot_backup_$(date +%Y%m%d).tar.gz trading_bot/

# Download backup to PC (from PC)
scp pi@192.168.1.XXX:~/trading_bot_backup_*.tar.gz ./

# Restore from backup (on Pi)
cd ~
tar -xzf trading_bot_backup_YYYYMMDD.tar.gz
```

### 🧹 Cleanup

```bash
# Clear old logs (keep last 1000 lines)
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT
tail -1000 autotrader.log > autotrader.log.tmp
mv autotrader.log.tmp autotrader.log

# Remove backup state files
rm autotrader_state.json.bak

# Check disk space
df -h
```

### 🔍 Troubleshooting

```bash
# Test Python environment
python3 -c "import pandas, numpy, flask; print('All imports OK')"

# Check Flask is installed
pip3 list | grep -i flask

# Verify API keys loaded
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key found' if os.getenv('API_KEY') else 'No API key')"

# Check port 5000 is listening
sudo netstat -tulpn | grep 5000

# Test network connectivity
ping -c 3 api.binance.com

# View system info
uname -a
python3 --version
```

### 📱 SSH from Phone

```bash
# Connect via SSH
ssh pi@192.168.1.XXX

# Quick status check
ssh pi@192.168.1.XXX "screen -ls && tail -5 ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT/autotrader.log"

# Restart everything remotely
ssh pi@192.168.1.XXX "screen -S trader -X quit && screen -S dashboard -X quit && cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT && ./start_pi.sh"
```

### 🌍 Dashboard Shortcuts

```bash
# From phone browser (same WiFi)
http://192.168.1.XXX:5000

# Bookmark this on your phone!
# Replace XXX with your actual Pi IP
```

### ⚡ Power Management

```bash
# Shutdown safely
sudo shutdown -h now

# Reboot
sudo reboot

# Schedule shutdown in 10 minutes
sudo shutdown -h +10

# Cancel scheduled shutdown
sudo shutdown -c
```

### 📈 Performance Monitoring

```bash
# CPU temperature
vcgencmd measure_temp

# Memory usage
free -h

# Disk usage
df -h

# Top processes
top

# Better top
htop
```

### 🔐 Security

```bash
# Change Pi password
passwd

# Update SSH config (disable password auth, use keys only)
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart ssh

# Check who's logged in
who

# View login history
last
```

---

## 🚨 Emergency Commands

```bash
# KILL EVERYTHING
screen -wipe
pkill -f autotrader
pkill -f dashboard

# FRESH START
cd ~/trading_bot/ML_OPTIMIZED_DEPLOYMENT
screen -dmS trader bash -c "python3 autotrader_ml_optimized.py"
screen -dmS dashboard bash -c "python3 dashboard.py"

# CHECK IF ANYTHING IS RUNNING
ps aux | grep -E "(autotrader|dashboard)" | grep -v grep
```

---

## 📞 Quick Reference URLs

- **Dashboard**: `http://YOUR_PI_IP:5000`
- **Pi Documentation**: `https://raspberrypi.com/documentation`
- **Screen Manual**: Run `man screen` on Pi
- **Python Docs**: `https://docs.python.org/3/`

---

## 💡 Pro Tips

1. **Bookmark dashboard on phone home screen** for instant access
2. **Use `screen -r`** to quickly check bot without stopping it
3. **Set up systemd** for auto-start on boot (production)
4. **Enable Tailscale** for remote access outside home network
5. **Check logs first** when troubleshooting issues
6. **Backup `.env` file** - it contains your API keys!
7. **Monitor temperature** - Pi should stay under 80°C
8. **Use UPS** (power backup) to prevent SD card corruption

---

## 📊 Dashboard Features Checklist

- [x] Real-time P&L tracking
- [x] Open positions with unrealized gains
- [x] Win rate & profit factor
- [x] Recent trade history (last 20)
- [x] Performance by symbol
- [x] Active cooldown timers
- [x] Live log streaming
- [x] Auto-refresh every 10 seconds
- [x] Mobile-optimized responsive design

---

**Print this page and keep it handy!** 📄

Most common use: SSH in → `screen -r trader` → check bot → Ctrl+A D → done!
