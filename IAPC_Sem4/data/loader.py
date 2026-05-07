from __future__ import annotations

from pathlib import Path

import pandas as pd

from IAPC_Sem4.config import DATA_DIR, PRICE_COLUMNS


def available_tickers(data_dir: Path = DATA_DIR) -> list[str]:
    return sorted(path.stem for path in data_dir.glob("*.csv"))


def load_data(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    missing = {"Date", *PRICE_COLUMNS}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.sort_values("Date").set_index("Date")
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].interpolate(method="time")
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].ffill().bfill()
    return df

