from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    look_back: int = 30
    train_fraction: float = 0.8
    meta_fraction_of_train: float = 0.2
    lstm_units: int = 64
    epochs: int = 20
    batch_size: int = 32
    random_state: int = 42
    rf_estimators: int = 100
    arima_refit_every: int = 20
    meta_min_rmse_improvement: float = 0.01
    meta_l2_strength: float = 0.0
    meta_max_weight: float = 0.70
    trading_selection_min_margin: float = 0.02
    trading_selection_min_trades: int = 4
    trading_selection_min_exposure: float = 5.0
    buy_threshold: float = 0.0003
    sell_threshold: float = -0.0003
    starting_cash: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    stacking_mode: str = "paper_like_in_sample"
    target_mode: str = "log_return"
