#!/bin/bash
# =============================================================================
# Quick Start Script - Raspberry Pi Trading Bot + Dashboard
# =============================================================================

echo "=========================================="
echo "🤖 Trading Bot + Dashboard Launcher"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 not found!${NC}"
    echo "Install with: sudo apt install python3 python3-pip"
    exit 1
fi

# Check if Flask is installed
if ! python3 -c "import flask" &> /dev/null; then
    echo -e "${YELLOW}Flask not found, installing...${NC}"
    pip3 install flask
fi

# Check if screen is installed (recommended for background processes)
if ! command -v screen &> /dev/null; then
    echo -e "${YELLOW}WARNING: 'screen' not installed${NC}"
    echo "Install with: sudo apt install screen"
    echo "This allows bot & dashboard to run in background"
    echo ""
fi

# Display menu
echo "Choose how to start:"
echo ""
echo "  1) Start bot only (foreground)"
echo "  2) Start dashboard only (foreground)"
echo "  3) Start both (foreground, 2 terminals recommended)"
echo "  4) Start both in background (using screen)"
echo "  5) View running screens"
echo "  6) Stop all background processes"
echo "  7) Exit"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo -e "${GREEN}Starting trading bot...${NC}"
        python3 autotrader_ml_optimized.py
        ;;
    2)
        echo -e "${GREEN}Starting dashboard...${NC}"
        echo ""
        echo "Access dashboard at:"
        echo "  Local:   http://localhost:5000"
        echo "  Network: http://$(hostname -I | awk '{print $1}'):5000"
        echo ""
        python3 dashboard.py
        ;;
    3)
        echo -e "${YELLOW}Starting in foreground - you need 2 terminals!${NC}"
        echo "Terminal 1: Run this script, choose option 1"
        echo "Terminal 2: Run this script, choose option 2"
        read -p "Press Enter to continue..."
        ;;
    4)
        if ! command -v screen &> /dev/null; then
            echo -e "${RED}ERROR: screen not installed${NC}"
            echo "Install with: sudo apt install screen"
            exit 1
        fi
        
        echo -e "${GREEN}Starting bot in background...${NC}"
        screen -dmS trader bash -c "cd $SCRIPT_DIR && python3 autotrader_ml_optimized.py"
        sleep 2
        
        echo -e "${GREEN}Starting dashboard in background...${NC}"
        screen -dmS dashboard bash -c "cd $SCRIPT_DIR && python3 dashboard.py"
        sleep 2
        
        echo ""
        echo -e "${GREEN}✓ Both services started in background${NC}"
        echo ""
        echo "View bot:       screen -r trader"
        echo "View dashboard: screen -r dashboard"
        echo "List all:       screen -ls"
        echo "Detach:         Ctrl+A then D"
        echo ""
        echo "Dashboard URL: http://$(hostname -I | awk '{print $1}'):5000"
        echo ""
        ;;
    5)
        echo ""
        echo "Running screen sessions:"
        screen -ls
        echo ""
        echo "To attach: screen -r <name>"
        echo "To detach: Ctrl+A then D"
        ;;
    6)
        echo -e "${YELLOW}Stopping background processes...${NC}"
        screen -S trader -X quit 2>/dev/null && echo "✓ Stopped trader"
        screen -S dashboard -X quit 2>/dev/null && echo "✓ Stopped dashboard"
        echo "Done"
        ;;
    7)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
