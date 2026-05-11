# data/loader.py

import os
import pandas as pd
from config import TICKER, START_DATE, END_DATE, DATA_DIR

def fetch_stock_data(ticker=TICKER, start=START_DATE, end=END_DATE, save=True):
    """Download OHLCV data from Yahoo Finance."""
    import yfinance as yf
    
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
    clean_ticker = ticker.replace('.', '_')
    candidates = [
        os.path.join(DATA_DIR, f"{clean_ticker}.csv"),
        os.path.join(DATA_DIR, f"{clean_ticker}_updated.csv"),
    ]
    path = next((candidate for candidate in candidates if os.path.exists(candidate)), candidates[0])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No local data found for {ticker}. Checked: {', '.join(candidates)}"
        )
    print(f"Loading local CSV -> {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df[list(required)] = df[list(required)].apply(pd.to_numeric, errors="coerce")
    df = df[list(required)].dropna()
    df.sort_index(inplace=True)
    return df

def get_data(ticker=TICKER, force_download=False):
    """
    Main entry point.
    Uses local CSV if available, downloads if not (or forced).
    """
    clean_ticker = ticker.replace('.', '_')
    path = os.path.join(DATA_DIR, f"{clean_ticker}.csv")
    updated_path = os.path.join(DATA_DIR, f"{clean_ticker}_updated.csv")
    if (os.path.exists(path) or os.path.exists(updated_path)) and not force_download:
        print(f"Loading local data for {ticker}...")
        return load_local_csv(ticker)
    return fetch_stock_data(ticker)
