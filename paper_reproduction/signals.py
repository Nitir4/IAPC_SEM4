from __future__ import annotations

import numpy as np
import pandas as pd


def compute_confidence(predictions: pd.Series, regime: str, volatility: float) -> str:
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


def compute_risk(regime: str, volatility: float, atr: float | None = None) -> str:
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

    if atr is not None and pd.notna(atr) and atr > 50:
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
    if ensemble_column not in predictions.columns:
        ensemble_column = model_columns[0]

    aligned_market = market_frame.reindex(predictions.index)
    out = predictions[model_columns].copy()
    out["ensemble"] = predictions[ensemble_column].astype(float)
    out["Signal"] = np.select(
        [out["ensemble"] > buy_threshold, out["ensemble"] < sell_threshold],
        ["BUY", "SELL"],
        default="HOLD",
    )

    if require_agreement and agreement_columns:
        agreement_frame = predictions[agreement_columns].astype(float)
        min_votes = max(1, min(agreement_min_votes, len(agreement_columns)))
        buy_votes = (agreement_frame > buy_threshold).sum(axis=1)
        sell_votes = (agreement_frame < sell_threshold).sum(axis=1)
        out.loc[(out["Signal"] == "BUY") & (buy_votes < min_votes), "Signal"] = "HOLD"
        out.loc[(out["Signal"] == "SELL") & (sell_votes < min_votes), "Signal"] = "HOLD"

    regimes = aligned_market["Regime"].fillna("Sideways")
    volatility = aligned_market["Volatility"].fillna(aligned_market.get("volatility_30"))
    atr = aligned_market["ATR"] if "ATR" in aligned_market else pd.Series(index=out.index, dtype=float)

    out["Confidence"] = [
        compute_confidence(
            predictions.loc[idx, agreement_columns],
            regimes.loc[idx],
            volatility.loc[idx],
        )
        for idx in out.index
    ]
    out["Risk"] = [
        compute_risk(regimes.loc[idx], volatility.loc[idx], atr.loc[idx])
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
    out["signal_num"] = out["Signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0}).astype(int)
    out["conf_num"] = out["Confidence"].map({"HIGH": 2, "MEDIUM": 1, "LOW": 0}).astype(int)
    out["risk_num"] = out["Risk"].map({"HIGH": 2, "MEDIUM": 1, "LOW": 0}).astype(int)
    return out
