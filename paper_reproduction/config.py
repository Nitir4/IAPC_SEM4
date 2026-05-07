from dataclasses import dataclass
import warnings


@dataclass(frozen=True)
class ExperimentConfig:
    look_back: int = 30
    train_fraction: float = 0.8
    meta_fraction_of_train: float = 0.25
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
    stacking_mode: str = "chronological_holdout"
    target_mode: str = "log_return"
    execution_mode: str = "next_open"

    def __post_init__(self):
        if self.execution_mode not in {'next_open','prior_close'}:
            raise ValueError('Execution must be next_open or prior_close.')
        if self.target_mode not in {'log_return','pct_change','price_change'}:
            raise ValueError('This regression/backtest pipeline supports return or price-change targets, not direction classification.')
        if not 0 < self.train_fraction < 1 or not 0 < self.meta_fraction_of_train < 1:
            raise ValueError('Training and meta-training fractions must lie strictly between zero and one.')
        if self.look_back < 1 or self.arima_refit_every < 1:
            raise ValueError('Look-back and ARIMA refit interval must be positive.')
        if self.stacking_mode == 'paper_like_in_sample':
            warnings.warn('paper_like_in_sample is an obsolete name; using chronological_holdout. No test-label fitting is performed.',UserWarning)
            object.__setattr__(self,'stacking_mode','chronological_holdout')
        elif self.stacking_mode != 'chronological_holdout':
            raise ValueError('Only chronological_holdout stacking is supported.')


def split_boundaries(n_rows, config):
    """One source of truth for actual integer split boundaries and reports."""
    meta_end = int(n_rows * config.train_fraction)
    base_end = int(meta_end * (1-config.meta_fraction_of_train))
    if base_end <= config.look_back + 4 or meta_end <= base_end or n_rows <= meta_end:
        raise ValueError('Not enough observations for indicator warm-up, training, meta training and testing.')
    return base_end, meta_end
