# plots/forecast_plots.py

import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_model_forecast(
    dates,
    actual,
    predicted,
    model_name,
    save_dir='outputs/plots'
):
    """
    Plot actual vs predicted returns for a model.
    """

    # ==========================================
    # Create output directory
    # ==========================================

    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # Build plot
    # ==========================================

    plt.figure(figsize=(14, 6))

    plt.plot(
        dates,
        actual,
        label='Actual Returns'
    )

    plt.plot(
        dates,
        predicted,
        label='Predicted Returns'
    )

    # ==========================================
    # Labels
    # ==========================================

    plt.title(f'{model_name} Forecast vs Actual')

    plt.xlabel('Date')

    plt.ylabel('Log Return')

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # ==========================================
    # Save
    # ==========================================

    filename = (
        f'{model_name.lower()}_forecast.png'
    )

    path = os.path.join(
        save_dir,
        filename
    )

    plt.savefig(path)

    plt.close()

    print(f'Saved forecast plot -> {path}')