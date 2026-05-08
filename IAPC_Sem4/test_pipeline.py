# test_pipeline.py  ← delete this after verification

from data.loader import get_data
from data.preprocess import compute_log_returns, add_forward_return, clean_data, split_data, get_features_target, scale_features
from data.indicators import add_indicators
from models.trainer import train_all_models, predict_all_models
from backtesting.bt_runner import run_backtest, compute_buy_and_hold

# 1. Load
df = get_data()
print("Raw shape:", df.shape)

# 2. Log returns
df = compute_log_returns(df)

# 3. Indicators
df = add_indicators(df)

# 4. Forward return target
df = add_forward_return(df)

# 5. Clean NaNs
df = clean_data(df)
print("Clean shape:", df.shape)
print("Columns:", df.columns.tolist())
from data.regime import detect_regime, regime_summary

# after clean_data line, add:
df = detect_regime(df)
regime_summary(df)
print("Regime sample:", df['Regime'].head(10).values)

# 6. Split
train, test = split_data(df)

# 7. Features/target
X_train, y_train = get_features_target(train)
X_test, y_test = get_features_target(test)

# 8. Scale
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

print("\nX_train shape:", X_train_sc.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test_sc.shape)
print("y_test shape:", y_test.shape)
print("\nSample targets:", y_train.head())
print("\nPipeline OK.")

print("\nTraining models...")
trained = train_all_models(X_train_sc, y_train)

print("\nPredicting...")
preds = predict_all_models(trained, X_test_sc)

for name, p in preds.items():
    print(f"{name}: first 3 preds = {p[:3].round(4)}")

from validation.overfit import run_overfit_analysis
from validation.walk_forward import walk_forward_validation

print("\n--- Overfitting Analysis ---")
overfit_results = run_overfit_analysis(trained, X_train_sc, y_train, X_test_sc, y_test)

print("\n--- Walk-Forward Validation ---")
wf_results = walk_forward_validation(df)

from signals.signal_generator import generate_signals

print("\n--- Signal Generation ---")
# Use test set predictions
signals_df = generate_signals(
    predictions_dict = preds,
    df_index         = y_test.index,
    regime_series    = df.loc[y_test.index, 'Regime'],
    volatility_series= df.loc[y_test.index, 'Volatility'],
    atr_series       = df.loc[y_test.index, 'ATR']
)

print(signals_df.head(10))
print("\nSignal distribution:")
print(signals_df['Signal'].value_counts())
print("Confidence distribution:")
print(signals_df['Confidence'].value_counts())
print("Risk distribution:")
print(signals_df['Risk'].value_counts())

from backtesting.bt_runner import run_backtest

print("\n--- Backtrader Backtest ---")
raw_df = df.loc[y_test.index]

metrics, cerebro, _ = run_backtest(
    ohlcv_df   = raw_df,
    signals_df = signals_df,
    printlog   = False
)

print("\n--- Buy & Hold Comparison ---")
bh = compute_buy_and_hold(raw_df)
print(f"  Strategy Return:   {metrics['Total Return %']}%")
print(f"  Buy & Hold Return: {bh}%")
gap = round(metrics['Total Return %'] - bh, 4)
print(f"  Alpha:             {gap}%")