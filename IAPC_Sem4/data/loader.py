# data/loader.py

import os
import yfinance as yf
import pandas as pd
from config import TICKER, START_DATE, END_DATE, DATA_DIR

def fetch_stock_data(ticker=TICKER, start=START_DATE, end=END_DATE, save=True):
    """Download OHLCV data from Yahoo Finance."""
    
    print(f"Fetching data for {ticker} from {start} to {end}...")
    
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True
    )

    # Fix yfinance multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)

        path = os.path.join(
            DATA_DIR,
            f"{ticker.replace('.', '_')}.csv"
        )

        df.to_csv(path)

        print(f"Saved to {path}")

    return df

def load_local_csv(ticker=TICKER):
    """Load previously saved local CSV."""
    path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No local data found at {path}. Run fetch_stock_data() first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.sort_index(inplace=True)
    return df

def get_data(ticker=TICKER, force_download=False):
    """
    Main entry point.
    Uses local CSV if available, downloads if not (or forced).
    """
    path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.csv")
    if os.path.exists(path) and not force_download:
        print(f"Loading local data for {ticker}...")
        return load_local_csv(ticker)
    return fetch_stock_data(ticker)