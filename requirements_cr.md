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

## Additional Market State Classification Logic

✅ 1. VARIABLE DEFINITIONS (Use in your Codex/Program)

Let:

GAS = Net_GEX_Above_Spot
GBS = Net_GEX_Below_Spot
DAS = Net_DEX_Above_Spot
DBS = Net_DEX_Below_Spot


Define signs:

sgn_GA = sign(GAS)      // +1 or –1
sgn_GB = sign(GBS)
sgn_DA = sign(DAS)
sgn_DB = sign(DBS)


Define magnitudes:

mag_GA = abs(GAS)
mag_GB = abs(GBS)
mag_DA = abs(DAS)
mag_DB = abs(DBS)

✅ 2. EQUATIONS TO IDENTIFY DOMINANT ZONES
2.1 Dominant GEX Location (Above or Below Spot)
if mag_GA > mag_GB:
    GEX_location = "ABOVE"
else:
    GEX_location = "BELOW"

2.2 Dominant DEX Location
if mag_DA > mag_DB:
    DEX_location = "ABOVE"
else:
    DEX_location = "BELOW"

2.3 Effective GEX Sign (Short or Long Gamma)
if GEX_location == "ABOVE":
    GEX_sign = sgn_GA
else:
    GEX_sign = sgn_GB

2.4 Effective DEX Sign (Direction Bias)
if DEX_location == "ABOVE":
    DEX_sign = sgn_DA
else:
    DEX_sign = sgn_DB

✅ 3. THE MASTER EQUATION FOR CLASSIFICATION
Your market state is uniquely identified by this ordered pair:
State = (GEX_location, GEX_sign, DEX_location, DEX_sign)


This generates all 12 core scenarios.

✅ 4. FORMAL RULES (12 CORE STATES)

These rules map directly to your matrix rows.

🔵 CASE GROUP 1 — GEX ABOVE SPOT
Rule 1:
if GEX_location == "ABOVE" and GEX_sign == +1 and DEX_location == "BELOW" and DEX_sign == +1:
    Scenario = "Best Bullish (Long Adam)"

Rule 2:
if GEX_location == "ABOVE" and GEX_sign == +1 and DEX_location == "BELOW" and DEX_sign == -1:
    Scenario = "Dip-Acceleration → Magnet Up (Conditional Long Adam)"

Rule 3:
if GEX_location == "ABOVE" and GEX_sign == +1 and DEX_location == "ABOVE" and DEX_sign == +1:
    Scenario = "Upside Stall (No Adam)"

Rule 4:
if GEX_location == "ABOVE" and GEX_sign == +1 and DEX_location == "ABOVE" and DEX_sign == -1:
    Scenario = "Low-Volatility Stall (Avoid Adam)"

🔴 CASE GROUP 2 — GEX BELOW SPOT
Rule 5:
if GEX_location == "BELOW" and GEX_sign == +1 and DEX_location == "BELOW" and DEX_sign == +1:
    Scenario = "Support + Weak Down Magnet (Weak Long Scalp)"

Rule 6:
if GEX_location == "BELOW" and GEX_sign == +1 and DEX_location == "BELOW" and DEX_sign == -1:
    Scenario = "Very Bearish (Strong Short Adam)"

Rule 7:
if GEX_location == "BELOW" and GEX_sign == +1 and DEX_location == "ABOVE" and DEX_sign == +1:
    Scenario = "Fade Rises (No Adam)"

Rule 8:
if GEX_location == "BELOW" and GEX_sign == +1 and DEX_location == "ABOVE" and DEX_sign == -1:
    Scenario = "Pop → Slam Down (Short Adam)"

🟣 CASE GROUP 3 — SHORT GAMMA ABOVE

(negative GEX above spot)

Rule 9:
if GEX_location == "ABOVE" and GEX_sign == -1 and DEX_location == "BELOW" and DEX_sign == +1:
    Scenario = "Bullish Explosion (Fast Long Adam)"

Rule 10:
if GEX_location == "ABOVE" and GEX_sign == -1 and DEX_location == "BELOW" and DEX_sign == -1:
    Scenario = "Volatility Whipsaw (Avoid Adam)"

Rule 11:
if GEX_location == "ABOVE" and GEX_sign == -1 and DEX_location == "ABOVE" and DEX_sign == +1:
    Scenario = "Uptrend + Brake (No Adam)"

Rule 12:
if GEX_location == "ABOVE" and GEX_sign == -1 and DEX_location == "ABOVE" and DEX_sign == -1:
    Scenario = "Short-Squeeze Blowout (Not Adam)"

⭐ 5. SPECIAL CASES (VERY IMPORTANT EQUATIONS)

These override the above rules when true.

⭐ Special Case A — Volatility Box
if sgn_GA == -1 and sgn_DB == -1:   # –GEX above AND –DEX below
    Scenario = "Volatility Box (Avoid)"

⭐ Special Case B — Dream Bullish
if sgn_GA == +1 and sgn_DB == +1:   # +GEX above AND +DEX below
    Scenario = "Dream Bullish (Perfect Long Adam)"

⭐ Special Case C — Super-Magnet Down
if (GAS < 0 and DAS < 0 and abs(GAS - DAS) < threshold):
    Scenario = "Negative–Negative Same Strike (Perfect Short Adam)"


Where threshold is a programmatic proximity rule:

threshold = 5% * max(abs(GAS), abs(DAS))

🚀 6. EQUATION FOR THE ZERO-LINE (GEX/DEX EQUALITY)
GEX Zero-Line Condition
if abs(GAS - GBS) < epsilon:
    GEX_zero = True

DEX Zero-Line Condition
if abs(DAS - DBS) < epsilon:
    DEX_zero = True

Regime Flip Trigger
if GEX_zero and DEX_zero:
    Trigger = "Regime Flip Zone (Vol Expansion → Adam Setup Soon)"


Where

epsilon = tolerance value (e.g., 2–5% of total gamma)

🔥 7. FINAL MARKET STATE EQUATION (FOR YOUR CODEX)
Market_State = f(GEX_location, GEX_sign, DEX_location, DEX_sign, special_cases, zero_line)
