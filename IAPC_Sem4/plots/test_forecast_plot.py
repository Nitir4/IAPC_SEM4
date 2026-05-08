# plots/test_forecast_plot.py

import pandas as pd

from forecast_plots import (
    plot_model_forecast
)

# ==========================================
# Load prediction CSV
# ==========================================

df = pd.read_csv(
    'outputs/predictions_AAPL.csv'
)

# ==========================================
# Convert Date column
# ==========================================

df['Date'] = pd.to_datetime(
    df['Date']
)

# ==========================================
# Models to plot
# ==========================================

models = [
    'linear',
    'ridge',
    'random_forest',
    'gradient_boost',
]

# ==========================================
# Generate plots
# ==========================================

for model in models:

    plot_model_forecast(
        dates=df['Date'],
        actual=df['Actual'],
        predicted=df[model],
        model_name=model
    )

print('\nAll forecast plots generated.')