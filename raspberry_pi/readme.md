# 📁 Raspberry Pi Setup Files

This folder contains everything you need to set up and run your trading bot on a Raspberry Pi with a web dashboard.

## 📂 Folder Structure

```
raspberry_pi/
├── docs/
│   ├── RASPBERRY_PI_COMPLETE_GUIDE.md    # Full setup walkthrough (START HERE)
│   ├── DASHBOARD_SETUP.md                # Dashboard installation & features
│   └── QUICK_REFERENCE.md                # Command cheat sheet
├── templates/
│   └── dashboard.html                    # Web dashboard interface
├── dashboard.py                          # Flask web server
└── start_pi.sh                          # Quick launcher script
```

## 🚀 Quick Start

1. **Read first**: [docs/RASPBERRY_PI_COMPLETE_GUIDE.md](docs/RASPBERRY_PI_COMPLETE_GUIDE.md)
2. **Transfer files** to your Raspberry Pi
3. **Run setup** commands from the guide
4. **Start dashboard**: `python3 dashboard.py`

## 📖 Documentation

- **[RASPBERRY_PI_COMPLETE_GUIDE.md](docs/RASPBERRY_PI_COMPLETE_GUIDE.md)** - Complete migration guide from Termux to Raspberry Pi (30+ pages)
- **[DASHBOARD_SETUP.md](docs/DASHBOARD_SETUP.md)** - Web dashboard features and installation
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Copy/paste command reference

## 🛠️ Files

- **dashboard.py** - Real-time web dashboard server (Flask)
- **start_pi.sh** - Interactive menu to start bot/dashboard
- **templates/dashboard.html** - Web interface (responsive, mobile-friendly)

## 📦 Installation on Raspberry Pi

```bash
# Transfer this entire folder to your Pi
scp -r raspberry_pi pi@YOUR_PI_IP:~/trading_bot/

# On the Pi
cd ~/trading_bot/raspberry_pi
pip3 install flask
python3 dashboard.py
```

Access dashboard at: `http://YOUR_PI_IP:5000`

## 💡 What This Gives You

✅ **24/7 stable operation** (no phone battery issues)  
✅ **SSH access** from anywhere  
✅ **Web dashboard** with live metrics  
✅ **Better performance** than phone  
✅ **Professional setup** with monitoring

---

**Total setup time**: ~2 hours  
**Hardware cost**: ~$85 (Raspberry Pi 4B + accessories)  
**Monthly cost**: ~$0.50 electricity

See [docs/RASPBERRY_PI_COMPLETE_GUIDE.md](docs/RASPBERRY_PI_COMPLETE_GUIDE.md) for complete instructions!
