# Methodology review — 2026-10-08

The user clarified that this is an original approach. The active `paper_reproduction/` directory name is retained for compatibility; published work is used to inform evaluation, rather than to claim replication.

## Published references

- Gu, Kelly and Xiu (2020), *Empirical Asset Pricing via Machine Learning*, **The Review of Financial Studies 33(5), 2223–2273**, DOI [10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009). The [published paper on the author's university site](https://dachxiu.chicagobooth.edu/download/ML.pdf), section 1.1, separates estimation, validation and testing; section 1.7 motivates squared-error comparison with a zero-return forecast. Those evaluation ideas inform our implementation. Their monthly excess-return panel and models differ from our per-ticker daily raw-return experiment; published results are not transferred here.
- Zhao and Cheng (2022), *Stock return prediction: Stacking a variety of models*, **Journal of Empirical Finance 67, 288–317**, DOI [10.1016/j.jempfin.2022.04.001](https://doi.org/10.1016/j.jempfin.2022.04.001). The [publisher abstract](https://www.sciencedirect.com/science/article/pii/S0927539822000342) discusses stacking stock-return models; its publication is also listed on [Cheng's research page](https://sites.google.com/site/tingtingcheng2014/research). This is related-method context only; the full text was not accessible in this pass.

These are published, relevant references. Publication does not establish that our model is correct or predictively useful.

## Corrected in this pass

| Issue | Current behavior | Evidence |
| --- | --- | --- |
| Whole-series interpolation/backfill could use future data | Past-only filling, leading incomplete rows discarded, warm-up indicators remain missing | Future-row perturbation and CSV validation tests |
| Regime and risk filters used the forecast day's realized data | Indicators are shifted one market observation before selecting regimes/signals | Same-day perturbations cannot change the forecast-day selection |
| Removed backfill exposed invalid neural training windows | Incomplete windows are excluded from training; invalid forecast windows raise an error | LSTM window/scaler tests using a model substitute |
| Configured split fractions were ignored | One shared boundary helper drives training and exports | Default/custom boundary tests and exported synthetic run |
| README described obsolete test-label fitting and a different model | Chronological holdout and active residual/ARMA protocol documented | Test-label perturbation leaves fitted residual predictions unchanged |
| First forecast-period return was lost | Returns are computed on the full market sequence before selecting the test region | Known-price accounting tests for both backtests |
| Execution prices, trade costs and initial drawdown disagreed | Prior-close execution, net round-trip results and initial-capital drawdown agree | Flat-price cost and persistent-HOLD tests |
| Missing signal dates could silently omit holding returns | Forecast dates must be consecutive observations | Gapped-window rejection tests |
| Second differencing was a two-period first difference | Repeated first differences are used | Quadratic-series differencing test |

## Validation boundary

The current regression suite and real-data acceptance results are recorded below. It includes an end-to-end runner/export/plot smoke test using real Random Forest and walk-forward ARMA on synthetic data. Most causal neural tests use a substitute; a separate acceptance test trains the real TensorFlow LSTM. The suite verifies causal information flow and controller/accounting behavior, not statistical superiority or suitability for live trading.

Historical output folders were preserved. Their metrics must not be represented as results of the corrected code.

## Remaining work

1. Document and preregister the original method, target, features, meta fitting, hyperparameters and chronological evaluation scheme before further model selection.
2. Extend the dated real-LSTM acceptance run to other tickers, periods and seeds with predeclared settings. Record dependencies, data hashes, usable observations and split boundaries.
3. Evaluate on an untouched holdout or predeclared rolling windows; avoid repeatedly choosing weights, thresholds or regime maps against the final test results.
4. Compare forecast errors to zero-return and lagged-return baselines, then assess statistical stability across tickers and periods. Confidence weighting currently expresses model agreement, not calibrated uncertainty.
5. Default execution now waits for the next open, with correct overnight holding returns. Extend sensitivity analysis to liquidity, delayed fills, consistent benchmark costs and adjustment/corporate-action policy. The legacy prior-close mode remains explicitly idealized.

Full scientific validation is in progress. These code corrections are a foundation for that work.

## Continued corrections

- Default backtesting now executes at the forecast day's observed open; entry, overnight holding and opening exits use distinct available prices. Both regular and persistent positions share the calculation, and legacy prior-close execution is opt-in.
- Return-based benchmark metrics and zero/lagged/historical-mean baseline forecasts are exported. An equal blend is included for comparison with learned residual weights. No baseline or hybrid is selected by final-test performance.
- Signal thresholds, agreement and confidence use consistent simple-return forecast units; price changes are normalized by the prior close, and log returns use expm1. Direction classification cannot silently be interpreted as a monetary return forecast.
- Undefined all-zero MAPE and invalid metric inputs are handled explicitly. Direction agreement treats zero as its own category.

## Executed acceptance — AAPL

The final suite contains **26 passing tests**, including one real TensorFlow LSTM fit/predict check. Separately, the full controller completed two real 20-epoch LSTM fits, real RF and walk-forward ARMA on the stored AAPL CSV (4,024 observations). The chronological regions contain 2,414 base-training rows, 805 meta-training rows and 805 test rows (2022-10-17 to 2025-12-31). Source/data hashes and package versions are recorded in the review workspace's `artifacts/iapc-aapl-20261008-training-manifest.json` and `artifacts/iapc-aapl-20261008/evaluation_summary.json`.

Frozen forecasts were reevaluated using the final signal/backtest code in both regular/persistent and next-open/prior-close modes. No model was retrained or selected against those final-test rankings.

| Forecast | RMSE on log returns | R2 relative to zero |
| --- | --- | --- |
| Zero forecast | 0.016863 | 0 |
| Training-prefix historical mean | 0.016841 | 0.002542 |
| ARMA | 0.016881 | -0.002121 |
| RF | 0.017589 | -0.087940 |
| LSTM | 0.017522 | -0.079665 |
| Residual hybrid | 0.017018 | -0.018469 |

**The residual hybrid did not beat the zero-return benchmark in this evaluated period.** These results establish execution and expose a modeling limitation; they do not justify changing hyperparameters against this already observed test window. Further evaluation needs predeclared additional windows/tickers/seeds. The stored adjusted-price provenance and retrospective corporate-action adjustments were not independently recertified. Confidence weighting and regime mapping remain heuristics.

## Follow-up self-financing correction

The corrected backtest now uses exact all-in proportional entry/exit accounting and a matched net buy-and-hold reference. Both use terminal mark-to-market without forced liquidation. The signal ATR filter is normalized by prior close, and silent invalid-strategy/signal fallbacks are rejected. See the external docs/IAPC_SEM4 implementation and walkthrough for current checks and a separately saved frozen-AAPL reevaluation; earlier numerical results in this document describe the earlier accounting convention. No final-test model selection or retraining was performed for this correction.
