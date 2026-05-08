# config.py

# --- Stock ---
TICKER = "RELIANCE.NS"        # change to whatever stock you're testing
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

# --- Paths ---
DATA_DIR = "data/raw/"
OUTPUT_DIR = "outputs/"

# --- Preprocessing ---
TARGET_COLUMN = "log_return"
TRAIN_RATIO = 0.8

# --- Indicators ---
SMA_WINDOWS = [10, 20, 50]
EMA_WINDOWS = [10, 20]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
ATR_PERIOD = 14
VOLATILITY_WINDOW = 20

# --- Regime ---
REGIME_SMA_FAST = 20
REGIME_SMA_SLOW = 50
VOLATILE_THRESHOLD = 0.02     # rolling std above this = Volatile

# --- Models ---
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 10
RF_N_ESTIMATORS = 200
GBM_MAX_DEPTH = 3
GBM_N_ESTIMATORS = 100
GBM_LEARNING_RATE = 0.05

# config.py — update these two lines only
BUY_THRESHOLD  = 0.0003  # predicted return > 0.03% → BUY
SELL_THRESHOLD = -0.0003  # predicted return < -0.03% → SELL

# --- Backtrader ---
STARTING_CASH = 100000
COMMISSION = 0.001            # 0.1%