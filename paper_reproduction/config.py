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
    stacking_mode: str = "paper_like_in_sample"
