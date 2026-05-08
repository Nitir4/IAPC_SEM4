# signals/risk_engine.py

def compute_risk(regime, volatility, atr=None):
    """
    Rule-based risk scoring.
    Returns: LOW / MEDIUM / HIGH
    """
    score = 0

    if regime == 'Volatile':
        score += 2
    elif regime == 'Bearish':
        score += 1

    if volatility is not None:
        if volatility > 0.025:
            score += 2
        elif volatility > 0.015:
            score += 1

    if atr is not None:
        if atr > 50:
            score += 1

    if score >= 4:
        return 'HIGH'
    elif score >= 2:
        return 'MEDIUM'
    else:
        return 'LOW'


def compute_risk_series(regime_series, volatility_series, atr_series=None):
    """Apply risk scoring row by row."""
    risks = []
    for idx in regime_series.index:
        regime = regime_series.loc[idx]
        vol    = volatility_series.loc[idx]
        atr    = atr_series.loc[idx] if atr_series is not None else None
        risks.append(compute_risk(regime, vol, atr))
    return risks