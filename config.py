"""
Configuration for ML Trading Strategy
"""

# Data Settings
ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT']
TIMEFRAME = '1h'
LOOKBACK_DAYS = 1095  # 3 years of data (more data = better patterns, less overfitting)

# Labeling Settings (RR-Based)
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5  # SL = entry_price - (1.5 × ATR) - Standard stop distance
RR_RATIO = 1.5  # TP = SL × 1.5 - Lower target for higher frequency (breakeven at 40% WR)
LOOKBACK_BARS = 100  # Max bars to check if TP/SL hit

# Feature Engineering
FEATURE_WINDOWS = [5, 10, 20, 50]  # Lookback periods for indicators
RSI_PERIODS = [7, 14, 21]
EMA_PERIODS = [9, 12, 20, 26, 50, 100, 200]
VOLUME_WINDOWS = [10, 20, 50]

# ML Training
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
RANDOM_SEED = 42
MIN_SAMPLES_FOR_TRAINING = 1000  # Minimum labeled samples needed

# Model Parameters
MODELS_TO_TRAIN = ['random_forest', 'xgboost', 'lightgbm']

# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_SPLIT = 50

# XGBoost
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 10
XGB_LEARNING_RATE = 0.05

# LightGBM
LGBM_N_ESTIMATORS = 200
LGBM_MAX_DEPTH = 10
LGBM_LEARNING_RATE = 0.05

# Backtesting
BACKTEST_INITIAL_CAPITAL = 10000
BACKTEST_RISK_PER_TRADE = 20  # $ risk per trade
BACKTEST_FEES = 0.001  # 0.1%
BACKTEST_SLIPPAGE = 0.0005  # 0.05%

# ML Strategy Parameters - Time-Enhanced Model
# Model: LightGBM_Robust_20260112_090713 (51 features with time awareness)
# Optimal threshold: 0.56 (ROC-AUC 0.565, ~1.4 trades/day, 52% win rate)
ML_PROBABILITY_THRESHOLD = 0.56
ML_ENTRY_FILTER = None
ML_USE_ATR_FROM_CONFIG = True  # Use ATR_MULTIPLIER and RR_RATIO from above
