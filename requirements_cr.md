# Change Request: Derived Regime Metrics for Options Analyzer

## Summary
This change request introduces additional derived columns and decision logic to enhance
regime classification and dealer bias scoring in the options analytics pipeline.

## Objectives
- Compute side-specific and aggregate Vanna-to-GEX ratios for calls and puts.
- Derive an optional relative strike distance normalization factor.
- Classify market regimes using deterministic rules informed by the new ratios and
  observed implied volatility direction.
- Score trade "energy" based on IV×OI magnitude compared with historical medians.
- Determine dealer bias signals in post-earnings contexts using updated ratios and
  volatility direction.

## Detailed Requirements

### 1. Required Inputs
The calculations rely on the following per-strike or aggregated fields, which must be
available in the analytics dataset prior to executing the new logic:

- `Call_Vanna`
- `Put_Vanna`
- `Call_GEX`
- `Put_GEX`
- `IVxOI` (implied volatility multiplied by open interest)
- `median_IVxOI` (benchmark median value for the current scope)
- `IV_Direction` (categorical: "up" or "down")
- `Strike`
- `Spot`

### 2. Derived Columns
Add the following computed metrics to the dataset:

| Column Name          | Formula                                                   | Notes |
|----------------------|------------------------------------------------------------|-------|
| `Call_Vanna_Ratio`   | `Call_Vanna / Call_GEX`                                    | Guard against divide-by-zero by returning `None` or `NaN` when `Call_GEX == 0`. |
| `Put_Vanna_Ratio`    | `Put_Vanna / Put_GEX`                                      | Guard against divide-by-zero by returning `None` or `NaN` when `Put_GEX == 0`. |
| `Vanna_GEX_Total`    | `(Call_Vanna + Put_Vanna) / (Call_GEX + Put_GEX)`          | Guard against divide-by-zero by returning `None` or `NaN` when denominator is zero. |
| `Rel_Dist` *(optional)* | `abs(Strike - Spot) / Spot`                             | Useful for normalization or bucketing by moneyness. |

All ratio calculations should support vectorized execution to maintain performance on large
option chains.

### 3. Core Logic Rules (Deterministic)

#### A. Regime Classification
Assign a `Regime` label using the decision tree below. The first matching rule in order of
precedence should be applied.

1. If `Call_Vanna_Ratio < 1` **and** `Put_Vanna_Ratio < 1`: set `Regime = "Gamma Pin"`.
2. Else if `Call_Vanna_Ratio > 2` **and** `IV_Direction == "up"`: set `Regime = "Pre-Earnings Fade"`.
3. Else if `Put_Vanna_Ratio > 2` **and** `IV_Direction == "down"`: set `Regime = "Post-Earnings Vanna Rally"`.
4. Else if `Call_Vanna_Ratio > 2` **and** `Put_Vanna_Ratio < 1`: set `Regime = "Vol Drift Down"`.
5. Else if `Call_Vanna_Ratio < 1` **and** `Put_Vanna_Ratio > 2`: set `Regime = "Vol Drift Up"`.
6. Else: set `Regime = "Transition Zone"`.

The classification must operate deterministically using the evaluated ratios and volatility
trend flag.

#### B. Energy / Magnitude Score
Calculate an `Energy_Score` from the IV×OI product relative to its median:

- If `IVxOI > 1.5 * median_IVxOI`: `Energy_Score = "High"`.
- Else if `IVxOI > 0.8 * median_IVxOI`: `Energy_Score = "Moderate"`.
- Otherwise: `Energy_Score = "Low"`.

`median_IVxOI` should be computed over the same scope as `IVxOI` (e.g., per expiry or ticker) and
must be available before this comparison.

#### C. Dealer Bias Score
Determine a `Dealer_Bias` signal, emphasizing post-earnings conditions where implied volatility
is falling:

- If `Put_Vanna_Ratio > 2` **and** `IV_Direction == "down"`: `Dealer_Bias = "Dealer Buying → Bullish Drift"`.
- Else if `Call_Vanna_Ratio > 2` **and** `IV_Direction == "down"`: `Dealer_Bias = "Dealer Selling → Bearish Fade"`.
- Otherwise: `Dealer_Bias = "Neutral / Mean Reversion"`.

### 4. Output Expectations
The pipeline must append the new derived columns and categorical labels to downstream
artifacts (JSON summaries, per-strike tables, dashboards). Documentation should be updated to
reflect the new metrics and decision logic.

### 5. Validation
- Unit tests should confirm ratio calculations handle zero denominators gracefully.
- Regression tests must verify regime, energy, and dealer bias outputs for representative
  combinations of ratios and volatility directions.
- Ensure optional `Rel_Dist` is present when strike-level data and spot price are available.

## Acceptance Criteria
- All specified derived columns and logic are implemented and accessible to downstream
  consumers.
- Documentation and tests demonstrate the behavior for key scenarios (gamma pin, pre- and
  post-earnings shifts, vol drift conditions, and neutral transitions).
- No regressions in existing analytics workflows.
