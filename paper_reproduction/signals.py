from __future__ import annotations

import numpy as np
import pandas as pd


def compute_confidence(predictions: pd.Series, regime: str, volatility: float) -> str:
    """Heuristic forecast-agreement label, not a calibrated success probability."""
    pred_std = float(predictions.std(ddof=0))
    if pred_std < 0.002:
        agreement_score = 3
    elif pred_std < 0.005:
        agreement_score = 2
    else:
        agreement_score = 1

    regime_penalty = 1 if regime == "Volatile" else 0
    volatility_penalty = 1 if pd.notna(volatility) and volatility > 0.02 else 0
    score = agreement_score - regime_penalty - volatility_penalty

    if score >= 3:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"


def compute_risk(regime: str, volatility: float, atr_fraction: float | None = None) -> str:
    """Heuristic scale-independent risk label; ATR is divided by prior close."""
    score = 0
    if regime == "Volatile":
        score += 2
    elif regime == "Bearish":
        score += 1

    if pd.notna(volatility):
        if volatility > 0.025:
            score += 2
        elif volatility > 0.015:
            score += 1

    if atr_fraction is not None and pd.notna(atr_fraction) and atr_fraction > 0.025:
        score += 1

    if score >= 4:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"


def generate_signals(
    predictions: pd.DataFrame,
    market_frame: pd.DataFrame,
    buy_threshold: float = 0.0003,
    sell_threshold: float = -0.0003,
    ensemble_column: str = "Hybrid",
    agreement_columns: list[str] | None = None,
    require_agreement: bool = False,
    agreement_min_votes: int = 2,
) -> pd.DataFrame:
    model_columns = [column for column in predictions.columns if column != "Actual"]
    if predictions.empty or not model_columns or ensemble_column not in model_columns:
        raise ValueError(f'Requested strategy {ensemble_column!r} needs a forecast column.')
    if not predictions.index.is_unique or not predictions.index.is_monotonic_increasing:
        raise ValueError('Forecast dates must be unique and chronological.')
    if not np.isfinite([buy_threshold,sell_threshold]).all() or sell_threshold >= buy_threshold:
        raise ValueError('Use finite, ordered sell and buy thresholds.')
    if agreement_columns is not None and not set(agreement_columns).issubset(model_columns):
        raise ValueError('Every requested agreement model needs a forecast column.')
    if agreement_columns is None:
        agreement_columns = [
            column
            for column in model_columns
            if not column.startswith("Naive")
            and column not in {"Hybrid", "EqualBlend", "TradingHybrid", "Sem4StyleEnsemble"}
        ]
    agreement_columns = [column for column in agreement_columns if column in predictions.columns]
    if not agreement_columns:
        agreement_columns = model_columns

    # Forecast row t is formed with information through t-1. Risk/confidence
    # filters must use that same information clock, not t's realized prices.
    aligned_market = market_frame.shift(1).reindex(predictions.index)
    target_mode=predictions.attrs.get('target_mode','log_return')
    signal_predictions=predictions[model_columns].astype(float).copy()
    if target_mode == 'log_return':
        signal_predictions=np.expm1(signal_predictions)
    elif target_mode == 'price_change':
        previous_close=aligned_market['Close']
        signal_predictions=signal_predictions.div(previous_close,axis=0)
    elif target_mode != 'pct_change':
        raise ValueError('Trading signals need log_return, pct_change or price_change forecasts.')
    if not np.isfinite(signal_predictions.to_numpy()).all():
        raise ValueError('Signal return forecasts must be finite and have a previous close.')
    out=predictions[model_columns].copy()
    out['ensemble']=signal_predictions[ensemble_column]
    out['SignalReturnUnits']='simple_return'
    out["Signal"] = np.select(
        [out["ensemble"] > buy_threshold, out["ensemble"] < sell_threshold],
        ["BUY", "SELL"],
        default="HOLD",
    )

    if require_agreement and agreement_columns:
        agreement_frame = signal_predictions[agreement_columns]
        if not isinstance(agreement_min_votes, (int, np.integer)) or not 1 <= agreement_min_votes <= len(agreement_columns):
            raise ValueError('Agreement vote count must lie between one and the model count.')
        min_votes = agreement_min_votes
        buy_votes = (agreement_frame > buy_threshold).sum(axis=1)
        sell_votes = (agreement_frame < sell_threshold).sum(axis=1)
        out.loc[(out["Signal"] == "BUY") & (buy_votes < min_votes), "Signal"] = "HOLD"
        out.loc[(out["Signal"] == "SELL") & (sell_votes < min_votes), "Signal"] = "HOLD"

    regimes = aligned_market["Regime"].fillna("Sideways")
    volatility = aligned_market["Volatility"]
    if "volatility_30" in aligned_market:
        volatility = volatility.fillna(aligned_market["volatility_30"])
    atr = aligned_market["ATR"] if "ATR" in aligned_market else pd.Series(index=out.index, dtype=float)
    atr_fraction = atr/aligned_market['Close']

    out["Confidence"] = [
        compute_confidence(
            signal_predictions.loc[idx, agreement_columns],
            regimes.loc[idx],
            volatility.loc[idx],
        )
        for idx in out.index
    ]
    out["Risk"] = [
        compute_risk(regimes.loc[idx], volatility.loc[idx], atr_fraction.loc[idx])
        for idx in out.index
    ]

    weak_buy = (out["Signal"] == "BUY") & (
        (out["Confidence"] == "LOW") | (out["Risk"] == "HIGH")
    )
    weak_sell = (out["Signal"] == "SELL") & (out["Confidence"] == "LOW")
    out.loc[weak_buy | weak_sell, "Signal"] = "HOLD"

    out["Regime"] = regimes
    out["Volatility"] = volatility
    out["ATR"] = atr
    out["ATRFraction"] = atr_fraction
    out["Agreement"] = out["Confidence"]
    out["signal_num"] = out["Signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0}).astype(int)
    out["conf_num"] = out["Confidence"].map({"HIGH": 2, "MEDIUM": 1, "LOW": 0}).astype(int)
    out["risk_num"] = out["Risk"].map({"HIGH": 2, "MEDIUM": 1, "LOW": 0}).astype(int)
    return out
