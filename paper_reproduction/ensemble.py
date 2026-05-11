from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

from .arima import rolling_arima_return_forecast
from .config import ExperimentConfig
from .data import build_target, random_forest_frame
from .lstm import fit_predict_lstm
from .metrics import regression_metrics
from .backtesting import run_strategy_backtests
from .signals import generate_signals


BASE_MODELS = ("ARIMA", "RandomForest", "LSTM")
BASE_MODEL_COLUMNS = list(BASE_MODELS)
META_MODEL_COLUMNS = list(BASE_MODELS)


def fit_constrained_meta(
    meta_x: pd.DataFrame,
    meta_y: np.ndarray,
    l2_strength: float = 0.0,
    max_weight: float = 0.70,
) -> np.ndarray:
    """Fit non-negative meta weights that sum to one."""
    x = meta_x[META_MODEL_COLUMNS].to_numpy(dtype=float)
    y = np.asarray(meta_y, dtype=float)
    n_models = len(META_MODEL_COLUMNS)
    prior = np.ones(n_models, dtype=float) / n_models
    upper_bound = min(max(max_weight, 1.0 / n_models), 1.0)

    def mse(weights: np.ndarray) -> float:
        fit_loss = float(np.mean((x @ weights - y) ** 2) * 1_000_000)
        l2_loss = float(l2_strength * np.sum((weights - prior) ** 2))
        return fit_loss + l2_loss

    result = minimize(
        mse,
        x0=np.ones(n_models) / n_models,
        method="SLSQP",
        bounds=[(0.0, upper_bound)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Constrained meta-learner failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def print_meta_validation_scores(meta_x: pd.DataFrame, meta_y: np.ndarray) -> None:
    rows = []
    for model_name in [*BASE_MODEL_COLUMNS, "NaiveZero", "NaiveLag1"]:
        rows.append((model_name, rmse(meta_y, meta_x[model_name].to_numpy(dtype=float))))
    rows.sort(key=lambda item: item[1])
    print("[meta validation] RMSE by candidate:", flush=True)
    for model_name, score in rows:
        print(f"  {model_name:<12} {score:.6f}", flush=True)


def apply_meta_safety_gate(
    meta_x: pd.DataFrame,
    meta_y: np.ndarray,
    weights: np.ndarray,
    min_rmse_improvement: float,
) -> tuple[np.ndarray, str, float]:
    hybrid_rmse = rmse(meta_y, meta_x[META_MODEL_COLUMNS].to_numpy(dtype=float) @ weights)
    zero_rmse = rmse(meta_y, meta_x["NaiveZero"].to_numpy(dtype=float))
    improvement = (zero_rmse - hybrid_rmse) / (zero_rmse + 1e-12)

    if min_rmse_improvement > 0 and improvement < min_rmse_improvement:
        safe_weights = np.ones(len(META_MODEL_COLUMNS), dtype=float) / len(META_MODEL_COLUMNS)
        return safe_weights, "EqualBlendFallback", improvement

    return weights, "LearnedMeta", improvement


def select_trading_strategy(
    validation_predictions: pd.DataFrame,
    market_frame: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[str, str, pd.DataFrame]:
    candidates = [*BASE_MODEL_COLUMNS, "Hybrid", "EqualBlend", "Sem4StyleEnsemble", "RegimeSwitching"]
    signal_frames = {
        candidate: generate_signals(
            predictions=validation_predictions,
            market_frame=market_frame,
            buy_threshold=config.buy_threshold,
            sell_threshold=config.sell_threshold,
            ensemble_column=candidate,
            agreement_columns=BASE_MODEL_COLUMNS,
        )
        for candidate in candidates
        if candidate in validation_predictions.columns
    }
    validation_backtests, _ = run_strategy_backtests(
        ohlcv_df=market_frame,
        signal_frames=signal_frames,
        starting_cash=config.starting_cash,
        commission=config.commission,
        slippage=config.slippage,
    )

    equal_row = validation_backtests.loc[
        validation_backtests["Strategy"] == "EqualBlend"
    ].iloc[0]
    eligible = validation_backtests[
        (validation_backtests["Total Trades"] >= config.trading_selection_min_trades)
        & (validation_backtests["Exposure %"] >= config.trading_selection_min_exposure)
    ]
    if eligible.empty:
        return "EqualBlend", "EqualBlendNoEligibleCandidate", validation_backtests

    best = eligible.sort_values("Final Value", ascending=False).iloc[0]
    required_value = float(equal_row["Final Value"]) * (1.0 + config.trading_selection_min_margin)
    if best["Strategy"] != "EqualBlend" and float(best["Final Value"]) >= required_value:
        return str(best["Strategy"]), "ValidationBacktestSelection", validation_backtests

    return "EqualBlend", "EqualBlendMarginFallback", validation_backtests


def ridge_weights(x: np.ndarray, y: np.ndarray, l2_strength: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if l2_strength <= 0:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    xtx = x.T @ x + l2_strength * np.eye(x.shape[1])
    return np.linalg.solve(xtx, x.T @ y)


def residual_hybrid_predict(
    meta_x: pd.DataFrame,
    meta_y: np.ndarray,
    test_x: pd.DataFrame,
    l2_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    arima_meta = meta_x["ARIMA"].to_numpy(dtype=float)
    residual_y = meta_y - arima_meta
    residual_features = np.column_stack(
        [
            meta_x["RandomForest"].to_numpy(dtype=float) - arima_meta,
            meta_x["LSTM"].to_numpy(dtype=float) - arima_meta,
        ]
    )
    weights = ridge_weights(residual_features, residual_y, l2_strength)

    arima_test = test_x["ARIMA"].to_numpy(dtype=float)
    residual_test = np.column_stack(
        [
            test_x["RandomForest"].to_numpy(dtype=float) - arima_test,
            test_x["LSTM"].to_numpy(dtype=float) - arima_test,
        ]
    )
    pred = arima_test + residual_test @ weights
    return pred, weights


def confidence_weighted_predict(test_x: pd.DataFrame, eps: float = 1e-6) -> np.ndarray:
    preds = []
    for _, row in test_x[BASE_MODEL_COLUMNS].iterrows():
        values = row.to_numpy(dtype=float)
        median = float(np.median(values))
        distances = np.abs(values - median)
        weights = 1.0 / (distances + eps)
        weights = weights / weights.sum()
        preds.append(float(weights @ values))
    return np.asarray(preds, dtype=float)





DEFAULT_REGIME_MAP = {
    "Bullish": "LSTM",
    "Bearish": "ARIMA",
    "Volatile": "RandomForest",
    "Sideways": "ARIMA",
}


def regime_switching_predict(
    test_x: pd.DataFrame,
    market_frame: pd.DataFrame,
    regime_map: dict[str, str] | None = None,
    default_model: str = "ARIMA",
) -> np.ndarray:
    if regime_map is None:
        regime_map = DEFAULT_REGIME_MAP
    regimes = market_frame.reindex(test_x.index)["Regime"].fillna("Sideways")
    preds = []
    for idx in test_x.index:
        regime = str(regimes.loc[idx])
        model = str(regime_map.get(regime, default_model))
        if model not in BASE_MODEL_COLUMNS:
            model = default_model
        preds.append(float(test_x.loc[idx, model]))
    return np.asarray(preds, dtype=float)


def sanity_check_predictions(
    predictions: pd.DataFrame,
    y_true: np.ndarray,
    label: str,
    z_threshold: float = 10.0,
) -> None:
    true_mean = float(np.mean(y_true))
    true_std = float(np.std(y_true))
    for column in predictions.columns:
        if column == "Actual":
            continue
        pred = predictions[column].to_numpy(dtype=float)
        z_scores = np.abs((pred - true_mean) / (true_std + 1e-8))
        max_z = float(np.max(z_scores))
        if max_z > z_threshold:
            print(
                f"  WARNING [{label}] {column}: max z-score={max_z:.1f}; "
                f"prediction range {pred.min():.2f} to {pred.max():.2f}, "
                f"actual range {y_true.min():.2f} to {y_true.max():.2f}",
                flush=True,
            )


def train_rf_predict(
    df: pd.DataFrame,
    train_end: int,
    predict_start: int,
    predict_end: int,
    config: ExperimentConfig,
    target_mode: str,
) -> np.ndarray:
    x, y = random_forest_frame(df, target_mode=target_mode)
    train_dates = df.index[:train_end]
    predict_dates = df.index[predict_start:predict_end]

    train_mask = x.index.isin(train_dates)
    predict_mask = x.index.isin(predict_dates)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=config.rf_estimators,
                    random_state=config.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x.loc[train_mask], y.loc[train_mask])
    return model.predict(x.loc[predict_mask])


def base_predictions(
    df: pd.DataFrame,
    target: pd.Series,
    train_end: int,
    predict_start: int,
    predict_end: int,
    config: ExperimentConfig,
    label: str,
    target_mode: str,
) -> tuple[pd.DataFrame, int]:
    steps = predict_end - predict_start
    print(f"[{label}] rolling ARMA forecast for {steps} return steps", flush=True)
    arima_pred, d = rolling_arima_return_forecast(
        target,
        train_end=train_end,
        predict_start=predict_start,
        predict_end=predict_end,
        refit_every=config.arima_refit_every,
    )
    print(f"[{label}] Random Forest forecast", flush=True)
    rf_pred = train_rf_predict(df, train_end, predict_start, predict_end, config, target_mode)
    print(f"[{label}] LSTM forecast ({config.epochs} epochs)", flush=True)
    lstm_pred = fit_predict_lstm(
        df=df,
        target=target,
        train_end=train_end,
        predict_start=predict_start,
        predict_end=predict_end,
        look_back=config.look_back,
        units=config.lstm_units,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.random_state,
    )

    predictions = pd.DataFrame(
        {
            "ARIMA": np.asarray(arima_pred, dtype=float),
            "RandomForest": np.asarray(rf_pred, dtype=float),
            "LSTM": np.asarray(lstm_pred, dtype=float),
            "NaiveZero": np.zeros(steps, dtype=float),
            "NaiveLag1": target.shift(1).iloc[predict_start:predict_end].to_numpy(dtype=float),
        },
        index=df.index[predict_start:predict_end],
    )
    return predictions, d


def run_ticker(
    ticker: str,
    df: pd.DataFrame,
    config: ExperimentConfig,
    target_mode: str = "log_return",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    n_rows = len(df)

    # ------------------------------------------------------------
    # DATA SPLITS
    #
    # 0%  -> 60% : Base-model training
    # 60% -> 80% : Meta-learner training (validation region)
    # 80% -> 100%: Final unseen test evaluation
    # ------------------------------------------------------------

    base_train_end = int(n_rows * 0.6)
    meta_train_end = int(n_rows * 0.8)
    target = build_target(df, mode=target_mode)

    # ============================================================
    # STEP 1:
    # Generate validation predictions for meta-learner training
    # ============================================================

    print(f"[{ticker}] generating meta-training predictions", flush=True)

    meta_x, meta_d = base_predictions(
        df=df,
        target=target,
        train_end=base_train_end,
        predict_start=base_train_end,
        predict_end=meta_train_end,
        config=config,
        label=f"{ticker} meta",
        target_mode=target_mode,
    )

    meta_y = target.iloc[base_train_end:meta_train_end].to_numpy(dtype=float)

    # ============================================================
    # STEP 2:
    # Generate FINAL TEST predictions
    # ============================================================

    print(f"[{ticker}] generating final test predictions", flush=True)

    test_x, test_d = base_predictions(
        df=df,
        target=target,
        train_end=meta_train_end,
        predict_start=meta_train_end,
        predict_end=n_rows,
        config=config,
        label=f"{ticker} test",
        target_mode=target_mode,
    )

    y_true = target.iloc[meta_train_end:n_rows].to_numpy(dtype=float)

    # ============================================================
    # STEP 3:
    # Ensemble predictions on UNSEEN test data
    # ============================================================

    residual_pred, residual_weights = residual_hybrid_predict(
        meta_x,
        meta_y,
        test_x,
        config.meta_l2_strength,
    )
    confidence_pred = confidence_weighted_predict(test_x)
    regime_pred = regime_switching_predict(test_x, df)

    # ============================================================
    # STEP 4:
    # Collect predictions
    # ============================================================

    keep_columns = [*BASE_MODEL_COLUMNS, "NaiveZero"]
    keep_columns = [column for column in keep_columns if column in test_x.columns]
    predictions = test_x[keep_columns].copy()

    predictions["ResidualHybrid"] = residual_pred
    predictions["ConfidenceWeight"] = confidence_pred
    predictions["RegimeSwitching"] = regime_pred
    predictions["Actual"] = y_true
    sanity_check_predictions(predictions, y_true, ticker)

    # ============================================================
    # STEP 5:
    # Metrics
    # ============================================================

    rows: list[dict[str, float | str | int]] = []

    model_names = [
        *BASE_MODELS,
        "NaiveZero",
        "ResidualHybrid",
        "ConfidenceWeight",
        "RegimeSwitching",
    ]

    for model_name in model_names:

        row = {
            "Ticker": ticker,
            "Model": model_name,
            **regression_metrics(
                y_true,
                predictions[model_name].to_numpy(),
            ),
        }

        rows.append(row)

    metrics = pd.DataFrame(rows)

    # ============================================================
    # Metadata
    # ============================================================

    metrics["ARIMA_d_meta"] = meta_d
    metrics["ARIMA_d_test"] = test_d
    metrics["target_mode"] = target_mode
    metrics["residual_weight_RandomForest"] = float(residual_weights[0])
    metrics["residual_weight_LSTM"] = float(residual_weights[1])

    metrics["base_train_fraction"] = 0.6
    metrics["meta_train_fraction"] = 0.2
    metrics["test_fraction"] = 0.2

    for key, value in asdict(config).items():
        metrics[key] = value

    return metrics, predictions
