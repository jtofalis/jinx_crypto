"""
Feature Engineering for ML Trading Strategy

Extracts 100+ features from OHLCV data for ML model training.
Each feature represents something the model can learn from.

Feature Categories:
1. Price-based: Momentum, returns, price position
2. Technical Indicators: RSI, MACD, Bollinger Bands, EMAs
3. Volume: Volume trends, surges, OBV
4. Market Structure: Volatility, ranges, trend strength
"""

import pandas as pd
import numpy as np
import warnings
from config import *

# Suppress DataFrame fragmentation warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def add_price_features(df):
    """
    Price-based features: returns, momentum, price position
    """
    # Returns over different periods
    for period in [1, 3, 5, 10, 20, 50]:
        df[f'return_{period}'] = df['close'].pct_change(period) * 100
    
    # High-Low ranges (volatility proxy)
    for period in [5, 10, 20]:
        df[f'hl_range_{period}'] = ((df['high'].rolling(period).max() - 
                                      df['low'].rolling(period).min()) / 
                                     df['close']) * 100
    
    # Close position within bar (bullish if close near high)
    df['close_position'] = ((df['close'] - df['low']) / 
                            (df['high'] - df['low'])).replace([np.inf, -np.inf], 0.5).fillna(0.5)
    
    # Body strength (how much of range is the body)
    df['body_strength'] = (abs(df['close'] - df['open']) / 
                           (df['high'] - df['low'])).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Upper/lower wick sizes
    df['upper_wick'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / 
                        (df['high'] - df['low'])).replace([np.inf, -np.inf], 0).fillna(0)
    df['lower_wick'] = ((df[['open', 'close']].min(axis=1) - df['low']) / 
                        (df['high'] - df['low'])).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def add_ema_features(df):
    """
    Exponential Moving Average features
    """
    # Calculate EMAs
    for period in EMA_PERIODS:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Price distance from EMAs (% above/below)
    for period in EMA_PERIODS:
        df[f'price_to_ema_{period}'] = ((df['close'] - df[f'ema_{period}']) / 
                                         df[f'ema_{period}']) * 100
    
    # EMA slopes (is EMA rising?)
    for period in [12, 26, 50]:
        df[f'ema_{period}_slope'] = df[f'ema_{period}'].pct_change(5) * 100
    
    # EMA crossovers
    df['ema_12_above_26'] = (df['ema_12'] > df['ema_26']).astype(int)
    df['ema_12_above_50'] = (df['ema_12'] > df['ema_50']).astype(int)
    df['ema_26_above_50'] = (df['ema_26'] > df['ema_50']).astype(int)
    
    return df

def add_rsi_features(df):
    """
    RSI (Relative Strength Index) features
    """
    for period in RSI_PERIODS:
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI slope (is RSI rising?)
        df[f'rsi_{period}_slope'] = df[f'rsi_{period}'].diff(3)
        
        # RSI position (oversold/overbought zones)
        df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
        df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
    
    return df

def add_macd_features(df):
    """
    MACD (Moving Average Convergence Divergence) features
    """
    # MACD calculation
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # MACD features
    df['macd_positive'] = (df['macd'] > 0).astype(int)
    df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
    df['macd_histogram_slope'] = df['macd_histogram'].diff(3)
    
    return df

def add_bollinger_bands_features(df):
    """
    Bollinger Bands features
    """
    # Calculate Bollinger Bands
    for period in [20]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df[f'bb_upper_{period}'] = sma + (std * 2)
        df[f'bb_middle_{period}'] = sma
        df[f'bb_lower_{period}'] = sma - (std * 2)
        
        # Price position relative to bands
        df[f'bb_position_{period}'] = ((df['close'] - df[f'bb_lower_{period}']) / 
                                        (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']))
        df[f'bb_position_{period}'] = df[f'bb_position_{period}'].replace([np.inf, -np.inf], 0.5).fillna(0.5)
        
        # Band width (volatility)
        df[f'bb_width_{period}'] = ((df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / 
                                     df[f'bb_middle_{period}']) * 100
        
        # Price touching bands
        df[f'price_at_lower_bb_{period}'] = (df['low'] <= df[f'bb_lower_{period}']).astype(int)
        df[f'price_at_upper_bb_{period}'] = (df['high'] >= df[f'bb_upper_{period}']).astype(int)
    
    return df

def add_stochastic_features(df):
    """
    Stochastic Oscillator features
    """
    period = 14
    
    # Stochastic K
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    df['stoch_k'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
    df['stoch_k'] = df['stoch_k'].replace([np.inf, -np.inf], 50).fillna(50)
    
    # Stochastic D (smoothed K)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Stochastic features
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_k_above_d'] = (df['stoch_k'] > df['stoch_d']).astype(int)
    
    return df

def add_volume_features(df):
    """
    Volume-based features
    """
    # Volume moving averages
    for period in VOLUME_WINDOWS:
        df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
    
    # Volume trend (is volume increasing?)
    df['volume_trend_5'] = df['volume'].rolling(window=5).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False
    )
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_slope'] = df['obv'].pct_change(10) * 100
    
    return df

def add_volatility_features(df):
    """
    Volatility and range features
    """
    # ATR already calculated in labeling
    if 'atr' in df.columns:
        # ATR relative to price
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # ATR trend
        df['atr_slope'] = df['atr'].pct_change(10) * 100
    
    # Historical volatility (standard deviation of returns)
    for period in [10, 20, 50]:
        df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std() * 100
    
    return df

def add_trend_features(df):
    """
    Trend strength and direction features
    """
    # ADX-like trend strength (simplified)
    for period in [14]:
        # Calculate directional movement
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Smooth
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()
        
        # Trend strength
        df[f'trend_strength_{period}'] = abs(plus_dm_smooth - minus_dm_smooth)
    
    # Higher highs / lower lows
    df['making_higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(window=5).sum()
    df['making_lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(window=5).sum()
    
    return df

def add_pattern_features(df):
    """
    Simple candlestick pattern features
    """
    # Bullish/bearish bars
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    
    # Consecutive bars in same direction
    df['consecutive_bullish'] = df['is_bullish'].rolling(window=5).sum()
    
    # Gap up/down
    df['gap_up'] = ((df['open'] > df['close'].shift(1)) & 
                    (df['open'] - df['close'].shift(1) > df['atr'] * 0.5)).astype(int)
    df['gap_down'] = ((df['open'] < df['close'].shift(1)) & 
                      (df['close'].shift(1) - df['open'] > df['atr'] * 0.5)).astype(int)
    
    return df

def add_time_features(df):
    """
    Add time-based features (hour, day of week, market sessions, liquidity)
    
    Assumes df has a datetime index or 'timestamp' column
    """
    # Ensure we have a datetime index
    if 'timestamp' in df.columns:
        df.index = pd.to_datetime(df['timestamp'])
    
    # Hour of day (0-23)
    df['hour'] = df.index.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df.index.dayofweek
    
    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Month start (first 3 days)
    df['is_month_start'] = (df.index.day <= 3).astype(int)
    
    # Month end (last 3 days)
    df['is_month_end'] = (df.index.day >= df.index.days_in_month - 2).astype(int)
    
    # Market sessions (UTC hour-based)
    # Asian session: 00:00-09:00 UTC
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
    
    # London session: 08:00-17:00 UTC
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
    
    # New York session: 13:00-22:00 UTC
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    
    # Session overlaps and cyclical features - use assign() to avoid fragmentation
    df = df.assign(
        london_ny_overlap=((df['hour'] >= 13) & (df['hour'] < 17)).astype(int),
        asian_london_overlap=((df['hour'] >= 8) & (df['hour'] < 9)).astype(int),
        # Liquidity score (0-3 based on session overlaps)
        liquidity_score=(df['asian_session'].astype(int) + 
                        df['london_session'].astype(int) + 
                        df['ny_session'].astype(int)),
        # Cyclical encoding for hour (sine/cosine to capture circularity)
        hour_sin=np.sin(2 * np.pi * df['hour'] / 24),
        hour_cos=np.cos(2 * np.pi * df['hour'] / 24),
        # Cyclical encoding for day of week
        day_sin=np.sin(2 * np.pi * df['day_of_week'] / 7),
        day_cos=np.cos(2 * np.pi * df['day_of_week'] / 7)
    )
    
    print(f"  Added time-based features: hour, day_of_week, sessions, liquidity")
    
    return df

def engineer_features(df):
    """
    Main function to add all features to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data and 'label' column
        
    Returns:
        DataFrame with all features added
    """
    print(f"Engineering features for {len(df)} bars...")
    
    # Add all feature groups
    df = add_price_features(df)
    df = add_ema_features(df)
    df = add_rsi_features(df)
    df = add_macd_features(df)
    df = add_bollinger_bands_features(df)
    df = add_stochastic_features(df)
    df = add_volume_features(df)
    df = add_volatility_features(df)
    df = add_trend_features(df)
    df = add_pattern_features(df)
    df = add_time_features(df)  # NEW: Time-based features
    
    # Drop rows with NaN (from rolling calculations)
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    print(f"  Added features, dropped {dropped} rows with NaN")
    print(f"  Final dataset: {len(df)} bars with {len(df.columns)} columns")
    
    return df

def prepare_ml_dataset(labeled_csv_path):
    """
    Load labeled CSV and add features
    
    Returns:
        X (features), y (labels)
    """
    # Load labeled data
    df = pd.read_csv(labeled_csv_path)
    
    # Add features
    df = engineer_features(df)
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in 
                   ['timestamp', 'label', 'open', 'high', 'low', 'close', 'volume']]
    
    X = df[feature_cols]
    y = df['label']
    
    return X, y, df

def main():
    """Test feature engineering on one asset"""
    print("="*80)
    print("FEATURE ENGINEERING TEST")
    print("="*80)
    
    # Load one labeled dataset
    csv_file = "BTC_USDT_labeled.csv"
    print(f"\nLoading: {csv_file}")
    
    X, y, df = prepare_ml_dataset(csv_file)
    
    print(f"\n{'='*80}")
    print("FEATURE ENGINEERING COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples: {len(X)}")
    print(f"Total features: {len(X.columns)}")
    print(f"Good entries: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"Bad entries: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
    
    print(f"\nFeature columns ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i}. {col}")
        if i >= 20:
            print(f"  ... and {len(X.columns) - 20} more features")
            break

if __name__ == '__main__':
    main()
