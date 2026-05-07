from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "IAPC_Sem4" / "outputs"

PRICE_COLUMNS = ["Close", "High", "Low", "Open", "Volume"]
TABULAR_FEATURE_COLUMNS = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_30",
    "MACD",
    "MACD_signal",
    "volatility_10",
    "volatility_30",
    "volume_avg_5",
    "volume_avg_20",
]
DEFAULT_TICKERS = ["AAPL", "GS", "JPM", "XOM", "TSLA"]
FIXED_ENSEMBLE_GROUPS = {
    "EnsembleGroupA": ("decision_tree", "linear", "svm"),
    "EnsembleGroupB": ("gradient_boost", "random_forest", "ridge"),
}


@dataclass(frozen=True)
class ExperimentConfig:
    train_fraction: float = 0.8
    random_state: int = 42
