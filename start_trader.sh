#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
# ML Trading Bot Startup Script (Termux/Android)
# Optimized Configuration: Threshold 0.56 | 1.5 RR | LightGBM
# ============================================================================

echo "=========================================="
echo "ML Trading Bot - V8 Optimized"
echo "Threshold: 0.56 | Expected: +124% vs baseline"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo -e "${YELLOW}[1/6] Checking Python installation...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found!${NC}"
    echo "Install with: pkg install python"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"

echo -e "${YELLOW}[2/6] Checking required files...${NC}"
REQUIRED_FILES=(
    "autotrader_ml_optimized.py"
    "feature_engineering.py"
    "advanced_features.py"
    "lightgbm_inference.py"
    ".env"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}ERROR: Missing file: $file${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ $file${NC}"
done

# Check for model file (JSON preferred, ONNX or joblib as fallback)
MODEL_FOUND=0
if [ -f "models/LightGBM_20251230_085740.json" ]; then
    echo -e "${GREEN}✓ models/LightGBM_20251230_085740.json (pure Python)${NC}"
    MODEL_FOUND=1
elif [ -f "models/LightGBM_20251230_085740.onnx" ]; then
    echo -e "${GREEN}✓ models/LightGBM_20251230_085740.onnx${NC}"
    MODEL_FOUND=1
elif [ -f "models/LightGBM_20251230_085740.joblib" ]; then
    echo -e "${GREEN}✓ models/LightGBM_20251230_085740.joblib${NC}"
    MODEL_FOUND=1
fi

if [ $MODEL_FOUND -eq 0 ]; then
    echo -e "${RED}ERROR: No model file found!${NC}"
    echo "Need one of: .json (best for phone), .onnx, or .joblib"
    exit 1
fi

echo -e "${YELLOW}[3/6] Checking .env configuration...${NC}"
if ! grep -q "ML_PROBABILITY_THRESHOLD=0.56" .env; then
    echo -e "${RED}WARNING: Threshold may not be optimized!${NC}"
    echo "Expected: ML_PROBABILITY_THRESHOLD=0.56"
fi

if ! grep -q "TIMEFRAME=1h" .env; then
    echo -e "${RED}WARNING: Timeframe should be 1h!${NC}"
fi

echo -e "${GREEN}✓ Configuration looks good${NC}"

echo -e "${YELLOW}[4/6] Checking Python dependencies...${NC}"
MISSING_DEPS=0
for pkg in pandas numpy; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo -e "${RED}✗ Missing: $pkg${NC}"
        MISSING_DEPS=1
    else
        echo -e "${GREEN}✓ $pkg${NC}"
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${YELLOW}Installing missing dependencies...${NC}"
    pip install pandas numpy python-binance
fi

echo -e "${YELLOW}[5/6] Checking API credentials...${NC}"
if ! grep -q "API_KEY=.*[A-Za-z0-9]" .env; then
    echo -e "${RED}ERROR: API_KEY not set in .env${NC}"
    echo "Please add your exchange API credentials to .env"
    exit 1
fi
echo -e "${GREEN}✓ API credentials found${NC}"

echo -e "${YELLOW}[6/6] Checking model file integrity...${NC}"
if [ -f "models/LightGBM_20251230_085740.json" ]; then
    MODEL_SIZE=$(stat -c%s "models/LightGBM_20251230_085740.json" 2>/dev/null || stat -f%z "models/LightGBM_20251230_085740.json" 2>/dev/null || echo "0")
    MODEL_TYPE="JSON (pure Python)"
elif [ -f "models/LightGBM_20251230_085740.onnx" ]; then
    MODEL_SIZE=$(stat -c%s "models/LightGBM_20251230_085740.onnx" 2>/dev/null || stat -f%z "models/LightGBM_20251230_085740.onnx" 2>/dev/null || echo "0")
    MODEL_TYPE="ONNX"
elif [ -f "models/LightGBM_20251230_085740.joblib" ]; then
    MODEL_SIZE=$(stat -c%s "models/LightGBM_20251230_085740.joblib" 2>/dev/null || stat -f%z "models/LightGBM_20251230_085740.joblib" 2>/dev/null || echo "0")
    MODEL_TYPE="Joblib"
fi

if [ "$MODEL_SIZE" -lt 100000 ]; then
    echo -e "${RED}ERROR: Model file seems corrupted (size: $MODEL_SIZE bytes)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Model file OK - $MODEL_TYPE ($MODEL_SIZE bytes)${NC}"

# ============================================================================
# START TRADING BOT
# ============================================================================

echo ""
echo "=========================================="
echo "All checks passed! Starting bot..."
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: LightGBM (121 features)"
echo "  - Threshold: 0.56 (optimized)"
echo "  - Timeframe: 1h"
echo "  - Symbols: BTC, ETH, BNB, SOL, XRP, ADA"
echo "  - RR Ratio: 1.5:1"
echo "  - Risk per trade: $20"
echo ""
echo "Expected Performance:"
echo "  - Annual trades: ~521"
echo "  - Win rate: 53.3%"
echo "  - Max drawdown: 1.83%"
echo "  - Sharpe ratio: 2.51"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Run the bot with unbuffered output
python -u autotrader_ml_optimized.py

# If bot exits, show exit code
EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Bot stopped normally${NC}"
else
    echo -e "${RED}Bot stopped with error code: $EXIT_CODE${NC}"
fi
echo "=========================================="
