"""Utilities for computing derived exposure metrics from unified datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Collection, Sequence

import pandas as pd

DERIVED_CSV_HEADER = [
    "Put VEX",
    "Put VEX Rank",
    "Call VEX",
    "Call VEX Rank",
    "Net VEX",
    "Net_DEX",
    "DEX_highlight",
    "Net_GEX",
    "Net_GEX_Highlight",
    "dGEX/dSpot",
    "dGEX/dSpot Rank",
    "Call_IVxOI",
    "Put_IVxOI",
    "IVxOI",
    "IVxOI Rank",
    "Call_IVxOI_Rank",
    "Put_IVxOI_Rank",
    "Top5_Regime_Energy_Bias",
    "Energy_Score",
    "Regime",
    "Dealer_Bias",
    "CoveredCall_Score",
    "CSP_Score",
    "Is_CC_Candidate",
    "Is_CSP_Candidate",
    "Strike",
    "Rel_Dist",
    "Call_Vanna",
    "Call_Vanna_Rank",
    "Put_Vanna",
    "Put_Vanna_Rank",
    "Net_Vanna",
    "Call_GEX",
    "Put_GEX",
    "Call_DEX",
    "Put_DEX",
    "Call_TEX",
    "Call_TEX_Highlight",
    "Put_TEX",
    "Put_TEX_Highlight",
    "Net_TEX",
    "TEX_highlight",
    "Call_IV",
    "Put_IV",
    "Median_IVxOI",
    "Call_Vanna_Ratio",
    "Put_Vanna_Ratio",
    "Vanna_GEX_Total",
    "Summary",
    "DateTime",
]

DERIVED_CSV_HEADER_TOP5_FIRST = [
    "Put VEX",
    "Put VEX Rank",
    "Call VEX",
    "Call VEX Rank",
    "Net VEX",
    "Net_DEX",
    "DEX_highlight",
    "Net_GEX",
    "Net_GEX_Highlight",
    "dGEX/dSpot",
    "dGEX/dSpot Rank",
    "Top5_Regime_Energy_Bias",
    "Energy_Score",
    "Regime",
    "Dealer_Bias",
    "CoveredCall_Score",
    "CSP_Score",
    "Is_CC_Candidate",
    "Is_CSP_Candidate",
    "Call_IVxOI",
    "Put_IVxOI",
    "IVxOI",
    "IVxOI Rank",
    "Call_IVxOI_Rank",
    "Put_IVxOI_Rank",
    "Strike",
    "Rel_Dist",
    "Call_Vanna",
    "Call_Vanna_Rank",
    "Put_Vanna",
    "Put_Vanna_Rank",
    "Net_Vanna",
    "Call_GEX",
    "Put_GEX",
    "Call_DEX",
    "Put_DEX",
    "Call_TEX",
    "Call_TEX_Highlight",
    "Put_TEX",
    "Put_TEX_Highlight",
    "Net_TEX",
    "TEX_highlight",
    "Call_IV",
    "Put_IV",
    "Median_IVxOI",
    "Call_Vanna_Ratio",
    "Put_Vanna_Ratio",
    "Vanna_GEX_Total",
    "Summary",
    "DateTime",
]

TOTAL_SUM_COLUMNS = {
    "Call_Vanna",
    "Put_Vanna",
    "Net_Vanna",
    "Call_GEX",
    "Put_GEX",
    "Net_GEX",
    "Call_DEX",
    "Put_DEX",
    "Net_DEX",
    "Net VEX",
    "Call_TEX",
    "Put_TEX",
    "Net_TEX",
    "Call_IVxOI",
    "Put_IVxOI",
    "IVxOI",
}

__all__ = [
    "DERIVED_CSV_HEADER",
    "apply_highlight_annotations",
    "classify_market_state",
    "compute_derived_metrics",
    "MarketState",
]


def _sign(value: float | None) -> int | None:
    if value is None or pd.isna(value):
        return None
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


MARKET_STATE_DESCRIPTIONS: dict[str, str] = {
    "Best Bullish (Long Adam)": (
        "Plain English: Market wants to go up. Dips get bought instantly. Very stable. "
        "Behavior: Slow grind up, shallow pullbacks, strong buy pressure. "
        "Adam: Best long setup of all."
    ),
    "Dip Acceleration → Magnet Up": (
        "Plain English: Dips fall faster than normal but bounce harder. "
        "Behavior: Quick flush → strong bounce → drift upward. "
        "Adam: Long only if the dip holds a key level."
    ),
    "Upside Stall (No Adam)": (
        "Plain English: Market tries to go up but keeps hitting invisible ceiling. "
        "Behavior: Slow, choppy grind with fadeable rips. "
        "Adam: Not safe — too slow, lacks momentum."
    ),
    "Low-Volatility Stall (Avoid Adam)": (
        "Plain English: Price goes nowhere. It's stuck in glue. "
        "Behavior: Small movements, chop, no trend. "
        "Adam: Avoid. No energy."
    ),
    "Support + Weak Magnet (Weak Long Scalp)": (
        "Plain English: There’s some support below, but market still leans downward. "
        "Behavior: Small bounces, but overall drifting down. "
        "Adam: Weak long scalp only — not a strong trend."
    ),
    "Very Bearish (Strong Short Adam)": (
        "Plain English: Market naturally wants to fall. Rallies fail fast. "
        "Behavior: Slide → bounce → deeper slide → breakdown. "
        "Adam: Excellent short setup."
    ),
    "Fade Rises (No Adam)": (
        "Plain English: Every pop gets sold. "
        "Behavior: Slow, controlled grind down. "
        "Adam: No — too controlled for momentum."
    ),
    "Pop → Slam Down (Short Adam)": (
        "Plain English: Market pops a bit, traps buyers, then slams down hard. "
        "Behavior: Quick move up → immediate reversal → big drop. "
        "Adam: Very good short setup."
    ),
    "Bullish Explosion (Fast Long Adam)": (
        "Plain English: Dealers are forced to buy on the way up. Explosive upside. "
        "Behavior: Fast upside breakout, vertical candles. "
        "Adam: Great for fast long entries."
    ),
    "Volatility Whipsaw (Avoid Adam)": (
        "Plain English: Both up and down moves accelerate → chaos. "
        "Behavior: Random violent swings. "
        "Adam: Avoid — unpredictable."
    ),
    "Uptrend With Brake (No Adam)": (
        "Plain English: Market goes up, but slowly and with hesitation. "
        "Behavior: Choppy, grindy up-move. "
        "Adam: No — too slow for momentum."
    ),
    "Short-Squeeze Blowout (Not Adam)": (
        "Plain English: Market explodes upward unpredictably. Too wild. "
        "Behavior: Violent vertical spikes. "
        "Adam: Avoid — impossible to place a safe stop."
    ),
    "Tug-of-War (No Adam)": (
        "Plain English: Bulls and bears cancel each other out and price chops. "
        "Behavior: Push up → pull back down repeatedly. "
        "Adam: Avoid until a clear winner emerges."
    ),
    "Super-Magnet Down (Strongest Short Adam)": (
        "Plain English: Gravity dominates and every rally snaps lower. "
        "Behavior: Relentless lower highs with hard flushes. "
        "Adam: Prime short setup until IV crush flips regime."
    ),
    "Fake Up → Drop (Short Scalp)": (
        "Plain English: Market fakes higher to lure buyers then dumps. "
        "Behavior: Quick pop into resistance then sharp rejection. "
        "Adam: Short scalp focused on the rejection."
    ),
    "Volatile Downtrend (Short Adam Possible)": (
        "Plain English: Downtrend with violent swings. "
        "Behavior: Spiky pullbacks that fail quickly. "
        "Adam: Short setups exist but require caution."
    ),
    "Volatility Box (Avoid)": (
        "Plain English: Market is wild and directionless. Both up and down moves "
        "accelerate. Behavior: Whipsaw → fake breakout → reverse → fake breakout "
        "again. Adam: Do NOT trade here."
    ),
    "Dream Bullish (Perfect Long Adam)": (
        "Plain English: Everything is aligned for a smooth, stable uptrend. "
        "Behavior: Dip → bounce → grind up for hours. "
        "Adam: The single most reliable long setup."
    ),
    "Negative–Negative Same Strike (Perfect Short Adam)": (
        "Plain English: There is no support and exposure pulls price straight down. "
        "Behavior: Pull → overshoot → failed base → continuation. "
        "Adam: One of the best short entries possible."
    ),
}


@dataclass(frozen=True)
class MarketStatePlaybook:
    next_step: str
    useful_metrics: tuple[str, ...]
    avoid: str


MARKET_STATE_PLAYBOOK: dict[str, MarketStatePlaybook] = {
    "Best Bullish (Long Adam)": MarketStatePlaybook(
        next_step="Look for Gamma Box bottom → prepare for long entry.",
        useful_metrics=("GEX curvature (dGEX/dSpot)", "VEX: positive VEX supports melt-up", "TEX: negative TEX confirms drift", "Expected Move & PC Walls → for targets"),
        avoid="Shorts. Overthinking. This is clean.",
    ),
    "Dip Acceleration → Magnet Up": MarketStatePlaybook(
        next_step="Wait for the dip → find Gamma Box support → long.",
        useful_metrics=("GEX distance below", "Relative distance to spot (Rel_Dist)", "VEX must not be aggressively negative", "IV direction: down = confirmation"),
        avoid="Chasing upside early.",
    ),
    "Upside Stall (No Adam)": MarketStatePlaybook(
        next_step=(
            "Check if stall is inside Gamma Box center. If yes → avoid. If at top of box → prepare fade. If breaking out → check VEX for confirmation"
        ),
        useful_metrics=("Gamma Box top boundary", "VEX direction (decides breakout vs stall)", "IVxOI congestion"),
        avoid="Directional Adam setups.",
    ),
    "Low-Volatility Stall (Avoid Adam)": MarketStatePlaybook(
        next_step="Pull Gamma Box and determine WHERE spot sits. If center → avoid everything. If bottom → watch for bounce. If top → watch for fade. If touching edge → breakout window",
        useful_metrics=("Gamma Box width", "dGEX/dSpot (flat = pinning)", "VEX: spike = end of stall"),
        avoid="Adam trades until a boundary interaction occurs.",
    ),
    "Support + Weak Magnet (Weak Long Scalp)": MarketStatePlaybook(
        next_step="Locate Gamma Box bottom → scalp long ONLY if VEX positive.",
        useful_metrics=("VEX: MUST be supportive", "Delta build-up (Net_DEX shrinking negative)", "TEX (theta-driven reversion)"),
        avoid="Expecting extended trend.",
    ),
    "Very Bearish (Strong Short Adam)": MarketStatePlaybook(
        next_step="Locate Gamma Box top → short the first failed rise.",
        useful_metrics=("dGEX/dSpot (negative slope = waterfall risk)", "VEX negativity (supercharges trend)", "Expected Move lower bound"),
        avoid="Going long unless regime flips.",
    ),
    "Fade Rises (No Adam)": MarketStatePlaybook(
        next_step="Find Gamma Box top → fade into resistance.",
        useful_metrics=("VEX shouldn’t be strongly positive", "dGEX/dSpot flat or slightly negative", "TEX supportive"),
        avoid="Breakout assumptions—this regime rarely breaks out cleanly.",
    ),
    "Pop → Slam Down (Short Adam)": MarketStatePlaybook(
        next_step="Wait for the pop into Gamma Box top → short.",
        useful_metrics=("VEX: strong negative = slam confirmation", "IV direction = up (fear) = stronger move", "PC Wall: if above spot → expect slam into wall"),
        avoid="Shorting too early before the pop.",
    ),
    "Bullish Explosion (Fast Long Adam)": MarketStatePlaybook(
        next_step="Confirm VEX positive then long the breakout.",
        useful_metrics=("Put VEX dominance (very bullish)", "dGEX/dSpot shaped upward", "Expected Move upper expansion"),
        avoid="Waiting too long — these runs go vertical.",
    ),
    "Volatility Whipsaw (Avoid Adam)": MarketStatePlaybook(
        next_step="Use Gamma Box to identify safety zones. Only scalp edges.",
        useful_metrics=("VEX spikes (avoid)", "IVxOI high clusters → boundaries", "dGEX/dSpot erratic"),
        avoid="Trend entries. Adam setups. Large positions.",
    ),
    "Uptrend With Brake (No Adam)": MarketStatePlaybook(
        next_step="Wait for Gamma Box breakout to remove the brake.",
        useful_metrics=("Gamma Box top", "VEX must flip positive", "dGEX/dSpot must steepen"),
        avoid="Trading inside the box.",
    ),
    "Short-Squeeze Blowout (Not Adam)": MarketStatePlaybook(
        next_step="Monitor for overextension → fade only at top-of-box extremes.",
        useful_metrics=("VEX positive (fuel)", "IV exploding = exhaustion signal", "EM deviation"),
        avoid="Directional Adam setups. Stops will be hit instantly.",
    ),
    "Tug-of-War (No Adam)": MarketStatePlaybook(
        next_step="Identify which force is building: VEX or TEX.",
        useful_metrics=("VEX (negative = bearish)", "TEX (positive = bullish drift)", "Gamma Box middle = chop zone"),
        avoid="Commitment until a force wins.",
    ),
    "Super-Magnet Down (Strongest Short Adam)": MarketStatePlaybook(
        next_step="Short ANY rise into Gamma Box top.",
        useful_metrics=("dGEX/dSpot → steep negative slope", "VEX negative", "PC Walls for targets", "Expected Move lower extension"),
        avoid="Longs unless major IV crush occurs.",
    ),
    "Fake Up → Drop (Short Scalp)": MarketStatePlaybook(
        next_step="Wait for fake-up → enter short at Gamma Box top.",
        useful_metrics=("VEX negative", "PC Wall overhead", "IV direction up (fear) helping down move"),
        avoid="Holding shorts too long; this is a scalp.",
    ),
    "Volatile Downtrend (Short Adam Possible)": MarketStatePlaybook(
        next_step="Look for pullbacks to Gamma Box upper boundary → short.",
        useful_metrics=("VEX negative", "dGEX/dSpot must not flatten", "IV rising = extra fuel"),
        avoid="Longs inside volatility spikes.",
    ),
}


@dataclass
class MarketState:
    scenario: str | None
    gex_location: str | None
    gex_sign: int | None
    dex_location: str | None
    dex_sign: int | None
    gex_zero: bool
    dex_zero: bool
    regime_flip: bool


@dataclass
class MarketState:
    scenario: str | None
    gex_location: str | None
    gex_sign: int | None
    dex_location: str | None
    dex_sign: int | None
    gex_zero: bool
    dex_zero: bool
    regime_flip: bool


def _format_timestamp(timestamp: datetime) -> str:
    utc_time = timestamp.astimezone(timezone.utc)
    return utc_time.replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def _validate_iv_direction(iv_direction: str) -> str:
    direction = (iv_direction or "").strip().lower()
    if direction not in {"up", "down", "unknown"}:
        raise ValueError("iv_direction must be 'up', 'down', or 'unknown'")
    return direction


def _format_strike(value: float) -> str:
    if pd.isna(value):
        return "NaN"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _classify_regime(row: pd.Series, iv_direction: str) -> str:
    call_ratio = row["Call_Vanna_Ratio"]
    put_ratio = row["Put_Vanna_Ratio"]

    def _is_less_than(value: float | pd.Series | None, threshold: float) -> bool:
        return pd.notna(value) and value < threshold

    def _is_greater_than(value: float | pd.Series | None, threshold: float) -> bool:
        return pd.notna(value) and value > threshold

    if _is_less_than(call_ratio, 1) and _is_less_than(put_ratio, 1):
        return "Gamma Pin"
    if _is_greater_than(call_ratio, 2) and iv_direction == "up":
        return "Pre-Earnings Fade"
    if _is_greater_than(put_ratio, 2) and iv_direction == "down":
        return "Post-Earnings Vanna Rally"
    if _is_greater_than(call_ratio, 2) and _is_less_than(put_ratio, 1):
        return "Vol Drift Down"
    if _is_less_than(call_ratio, 1) and _is_greater_than(put_ratio, 2):
        return "Vol Drift Up"
    return "Transition Zone"


def classify_market_state(
    net_gex_above_spot: float | None,
    net_gex_below_spot: float | None,
    net_dex_above_spot: float | None,
    net_dex_below_spot: float | None,
    *,
    epsilon: float | None = None,
) -> MarketState:
    """Classify the market state based on GEX/DEX distributions."""

    if None in {
        net_gex_above_spot,
        net_gex_below_spot,
        net_dex_above_spot,
        net_dex_below_spot,
    }:
        return MarketState(None, None, None, None, None, False, False, False)

    def _sign(value: float) -> int:
        return 1 if value >= 0 else -1

    gas = float(net_gex_above_spot)
    gbs = float(net_gex_below_spot)
    das = float(net_dex_above_spot)
    dbs = float(net_dex_below_spot)

    sgn_ga = _sign(gas)
    sgn_gb = _sign(gbs)
    sgn_da = _sign(das)
    sgn_db = _sign(dbs)

    mag_ga = abs(gas)
    mag_gb = abs(gbs)
    mag_da = abs(das)
    mag_db = abs(dbs)

    gex_location = "ABOVE" if mag_ga > mag_gb else "BELOW"
    dex_location = "ABOVE" if mag_da > mag_db else "BELOW"

    gex_sign = sgn_ga if gex_location == "ABOVE" else sgn_gb
    dex_sign = sgn_da if dex_location == "ABOVE" else sgn_db

    # Special cases override standard rules.
    scenario: str | None = None
    if sgn_ga == -1 and sgn_db == -1:
        scenario = "Volatility Box (Avoid)"
    elif sgn_ga == 1 and sgn_db == 1:
        scenario = "Dream Bullish (Perfect Long Adam)"
    else:
        threshold = 0.05 * max(mag_ga, mag_da)
        if gas < 0 and das < 0 and abs(gas - das) < threshold:
            scenario = "Negative–Negative Same Strike (Perfect Short Adam)"

    if scenario is None:
        key = (gex_location, gex_sign, dex_location, dex_sign)
        match key:
            case ("ABOVE", 1, "BELOW", 1):
                scenario = "Best Bullish (Long Adam)"
            case ("ABOVE", 1, "BELOW", -1):
                scenario = "Dip Acceleration → Magnet Up"
            case ("ABOVE", 1, "ABOVE", 1):
                scenario = "Upside Stall (No Adam)"
            case ("ABOVE", 1, "ABOVE", -1):
                scenario = "Low-Volatility Stall (Avoid Adam)"
            case ("BELOW", 1, "BELOW", 1):
                scenario = "Support + Weak Magnet (Weak Long Scalp)"
            case ("BELOW", 1, "BELOW", -1):
                scenario = "Very Bearish (Strong Short Adam)"
            case ("BELOW", 1, "ABOVE", 1):
                scenario = "Fade Rises (No Adam)"
            case ("BELOW", 1, "ABOVE", -1):
                scenario = "Pop → Slam Down (Short Adam)"
            case ("ABOVE", -1, "BELOW", 1):
                scenario = "Bullish Explosion (Fast Long Adam)"
            case ("ABOVE", -1, "BELOW", -1):
                scenario = "Volatility Whipsaw (Avoid Adam)"
            case ("ABOVE", -1, "ABOVE", 1):
                scenario = "Uptrend With Brake (No Adam)"
            case ("ABOVE", -1, "ABOVE", -1):
                scenario = "Short-Squeeze Blowout (Not Adam)"
            case _:
                scenario = None

    total_gex = mag_ga + mag_gb
    total_dex = mag_da + mag_db
    effective_total = max(total_gex, total_dex, 1.0)
    epsilon_value = epsilon if epsilon is not None else 0.05 * effective_total

    gex_zero = abs(gas - gbs) < epsilon_value
    dex_zero = abs(das - dbs) < epsilon_value
    regime_flip = gex_zero and dex_zero

    return MarketState(
        scenario,
        gex_location,
        gex_sign,
        dex_location,
        dex_sign,
        gex_zero,
        dex_zero,
        regime_flip,
    )


def _score_energy(ivxoi: float | pd.Series, median_ivxoi: float | None) -> str:
    if pd.isna(ivxoi) or median_ivxoi is None or pd.isna(median_ivxoi):
        return "Low"
    if median_ivxoi <= 0:
        return "High" if ivxoi > 0 else "Low"
    if ivxoi > 1.5 * median_ivxoi:
        return "High"
    if ivxoi > 0.8 * median_ivxoi:
        return "Moderate"
    return "Low"


def _dealer_bias(row: pd.Series, iv_direction: str) -> str:
    call_ratio = row["Call_Vanna_Ratio"]
    put_ratio = row["Put_Vanna_Ratio"]

    call_gt_two = pd.notna(call_ratio) and call_ratio > 2
    put_gt_two = pd.notna(put_ratio) and put_ratio > 2

    if put_gt_two and iv_direction == "down":
        return "Dealer Buying → Bullish Drift"
    if call_gt_two and iv_direction == "down":
        return "Dealer Selling → Bearish Fade"
    return "Neutral / Mean Reversion"


def _bias_signal(energy_score: str, regime: str) -> str:
    energy = (energy_score or "").strip().lower()
    scenario = (regime or "").strip().lower()

    if energy != "high":
        return ""

    if scenario == "gamma pin":
        return "FADE"
    if scenario == "vol drift up":
        return "LONG"
    if scenario == "vol drift down":
        return "SHORT"
    if scenario == "transition zone":
        return "WAIT FOR BREAKOUT"
    return ""


def apply_highlight_annotations(metrics: pd.DataFrame) -> None:
    """Populate highlight columns for the most notable strike metrics."""

    highlight_columns = [
        "Call_Vanna_Rank",
        "Put_Vanna_Rank",
        "Net_GEX_Highlight",
        "DEX_highlight",
        "Call_TEX_Highlight",
        "Put_TEX_Highlight",
        "TEX_highlight",
        "Call_IVxOI_Rank",
        "Put_IVxOI_Rank",
    ]

    for column in highlight_columns:
        if column not in metrics.columns:
            metrics[column] = ""
        else:
            metrics[column] = ""

    if metrics.empty:
        return

    def _strike_label(idx: object) -> str:
        if "Strike" in metrics.columns:
            strike_value = metrics.at[idx, "Strike"]
        else:
            strike_value = idx
        strike_numeric = pd.to_numeric(
            pd.Series([strike_value], dtype="object"), errors="coerce"
        ).iloc[0]
        return _format_strike(strike_numeric)

    def _set_ranked_highlights(
        value_column: str,
        highlight_column: str,
        top_n: int,
        *,
        use_nsmallest: bool = False,
        label: str = "Top",
        exclude_indices: Collection | None = None,
    ) -> list:
        if value_column not in metrics.columns or highlight_column not in metrics.columns:
            return []

        series = pd.to_numeric(metrics[value_column], errors="coerce")
        series = series.dropna()
        if series.empty:
            return []

        limit = min(top_n, len(series))
        ranked_indices = (
            series.nsmallest(limit).index if use_nsmallest else series.nlargest(limit).index
        )
        excluded = set(exclude_indices or [])
        used_indices: list = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if idx in excluded:
                continue
            strike_value = _strike_label(idx)
            metrics.at[idx, highlight_column] = f"{label} {rank} : {strike_value}"
            used_indices.append(idx)
        return used_indices

    _set_ranked_highlights("Call_Vanna", "Call_Vanna_Rank", top_n=4)
    _set_ranked_highlights(
        "Put_Vanna",
        "Put_Vanna_Rank",
        top_n=4,
        use_nsmallest=True,
    )
    top_net_gex = _set_ranked_highlights("Net_GEX", "Net_GEX_Highlight", top_n=4)
    _set_ranked_highlights(
        "Net_GEX",
        "Net_GEX_Highlight",
        top_n=4,
        use_nsmallest=True,
        label="Bottom",
        exclude_indices=top_net_gex,
    )
    top_net_dex = _set_ranked_highlights("Net_DEX", "DEX_highlight", top_n=4)
    _set_ranked_highlights(
        "Net_DEX",
        "DEX_highlight",
        top_n=4,
        use_nsmallest=True,
        label="Bottom",
        exclude_indices=top_net_dex,
    )
    _set_ranked_highlights(
        "Call_TEX",
        "Call_TEX_Highlight",
        top_n=5,
        use_nsmallest=True,
    )
    _set_ranked_highlights(
        "Put_TEX",
        "Put_TEX_Highlight",
        top_n=5,
        use_nsmallest=True,
    )
    _set_ranked_highlights(
        "Net_TEX",
        "TEX_highlight",
        top_n=5,
        use_nsmallest=True,
    )
    _set_ranked_highlights("Call_IVxOI", "Call_IVxOI_Rank", top_n=4)
    _set_ranked_highlights("Put_IVxOI", "Put_IVxOI_Rank", top_n=4)


def compute_derived_metrics(
    unified_df: pd.DataFrame,
    *,
    calculation_time: datetime,
    spot_price: float | None = None,
    iv_direction: str = "down",
    drop_columns: Sequence[str] | None = None,
    include_totals_row: bool = False,
    include_put_vex: bool = True,
    append_market_state_row: bool = False,
) -> pd.DataFrame:
    """Return the Phase 1 derived exposure metrics for ``unified_df``."""

    if "Strike" not in unified_df.columns:
        raise KeyError("Unified dataset must contain a 'Strike' column")

    timestamp_str = _format_timestamp(calculation_time)

    required_columns = {
        "call_vanna",
        "puts_vanna",
        "net_vanna",
        "call_gex",
        "puts_gex",
        "net_gex",
        "call_gamma",
        "puts_gamma",
        "call_open_interest",
        "puts_open_interest",
        "call_delta",
        "puts_delta",
        "call_theta",
        "puts_theta",
        "call_iv",
        "puts_iv",
        "call_oi_iv",
        "puts_oi_iv",
    }
    missing_columns = required_columns.difference(unified_df.columns)
    if missing_columns:
        raise KeyError(
            "Unified dataset missing expected columns: "
            + ", ".join(sorted(missing_columns))
        )

    direction_value = _validate_iv_direction(iv_direction)

    call_open_interest = pd.to_numeric(
        unified_df["call_open_interest"], errors="coerce"
    ).astype(float)
    put_open_interest = pd.to_numeric(
        unified_df["puts_open_interest"], errors="coerce"
    ).astype(float)
    call_gamma = pd.to_numeric(unified_df["call_gamma"], errors="coerce").astype(float)
    put_gamma = pd.to_numeric(unified_df["puts_gamma"], errors="coerce").astype(float)

    metrics = pd.DataFrame(
        {
            "Strike": unified_df["Strike"].astype(float),
            "Call_Vanna": unified_df["call_vanna"].astype(float),
            "Put_Vanna": unified_df["puts_vanna"].astype(float),
            "Net_Vanna": unified_df["net_vanna"].astype(float),
            "Call_GEX": unified_df["call_gex"].astype(float),
            "Put_GEX": unified_df["puts_gex"].astype(float),
            "Net_GEX": unified_df["net_gex"].astype(float),
            "Call_DEX": (
                unified_df["call_delta"].astype(float)
                * call_open_interest
                * 100
            ),
            "Put_DEX": (
                unified_df["puts_delta"].astype(float)
                * put_open_interest
                * 100
            ),
            "Call_TEX": (
                unified_df["call_theta"].astype(float)
                * call_open_interest
                * 100
            ),
            "Put_TEX": (
                unified_df["puts_theta"].astype(float)
                * put_open_interest
                * 100
            ),
            "Call_IV": unified_df["call_iv"].astype(float).round(1),
            "Put_IV": unified_df["puts_iv"].astype(float).round(1),
            "Call_IVxOI": unified_df["call_oi_iv"].astype(float).round(1),
            "Put_IVxOI": unified_df["puts_oi_iv"].astype(float).round(1),
        }
    )

    optional_candidate_columns = {
        "CoveredCall_Score": 0.0,
        "CSP_Score": 0.0,
        "Is_CC_Candidate": False,
        "Is_CSP_Candidate": False,
    }

    for column, default_value in optional_candidate_columns.items():
        if column in unified_df.columns:
            filled = unified_df[column]
            if filled.dtype == "object":
                filled = filled.convert_dtypes()
            filled = filled.fillna(default_value)
            metrics[column] = filled
        else:
            metrics[column] = default_value

    if include_put_vex:
        put_vex_values = (
            metrics["Put_Vanna"].astype(float)
            * metrics["Put_IV"].astype(float)
            * put_open_interest
        ).round(2)

        metrics.insert(0, "Put VEX", put_vex_values)
        metrics.insert(1, "Put VEX Rank", "")

        call_vex_values = (
            metrics["Call_Vanna"].astype(float)
            * metrics["Call_IV"].astype(float)
            * call_open_interest
        ).round(2)

        metrics.insert(2, "Call VEX", call_vex_values)
        metrics.insert(3, "Call VEX Rank", "")

        net_vex_values = (call_vex_values + put_vex_values).round(2)

        metrics.insert(4, "Net VEX", net_vex_values)

        negative_put_vex = put_vex_values[put_vex_values < 0]
        if not negative_put_vex.empty:
            ranked_indices = negative_put_vex.nsmallest(
                min(7, len(negative_put_vex))
            ).index
            for rank, idx in enumerate(ranked_indices, start=1):
                strike_value = _format_strike(metrics.at[idx, "Strike"])
                metrics.at[idx, "Put VEX Rank"] = f"Rank {rank}: {strike_value}"

        positive_call_vex = call_vex_values[call_vex_values > 0]
        if not positive_call_vex.empty:
            ranked_indices = positive_call_vex.nlargest(min(5, len(positive_call_vex))).index
            for rank, idx in enumerate(ranked_indices, start=1):
                strike_value = _format_strike(metrics.at[idx, "Strike"])
                metrics.at[idx, "Call VEX Rank"] = f"Rank {rank}: {strike_value}"

    metrics["Net_DEX"] = metrics["Call_DEX"] + metrics["Put_DEX"]
    metrics["Net_TEX"] = metrics["Call_TEX"] + metrics["Put_TEX"]

    for column in ("Call_DEX", "Put_DEX", "Net_DEX"):
        metrics[column] = metrics[column].round(1)

    for column in ("Call_TEX", "Put_TEX", "Net_TEX"):
        metrics[column] = metrics[column].round(1)

    metrics["IVxOI"] = (
        metrics[["Call_IVxOI", "Put_IVxOI"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
        .round(1)
    )
    metrics["IVxOI Rank"] = ""

    ivxoi_values = pd.to_numeric(metrics["IVxOI"], errors="coerce")
    ivxoi_rank_series = ivxoi_values[ivxoi_values > 0].dropna()
    if not ivxoi_rank_series.empty:
        ranked_indices = ivxoi_rank_series.nlargest(min(7, len(ivxoi_rank_series))).index
        for rank, idx in enumerate(ranked_indices, start=1):
            strike_value = _format_strike(metrics.at[idx, "Strike"])
            metrics.at[idx, "IVxOI Rank"] = f"Rank {rank}: {strike_value}"

    call_gex_denom = metrics["Call_GEX"].replace({0: pd.NA})
    put_gex_denom = metrics["Put_GEX"].replace({0: pd.NA})

    metrics["Call_Vanna_Ratio"] = metrics["Call_Vanna"] / call_gex_denom
    metrics["Put_Vanna_Ratio"] = metrics["Put_Vanna"] / put_gex_denom
    total_gex_denom = (metrics["Call_GEX"] + metrics["Put_GEX"]).replace({0: pd.NA})
    metrics["Vanna_GEX_Total"] = metrics["Net_Vanna"] / total_gex_denom

    metrics["Call_Vanna_Ratio"] = metrics["Call_Vanna_Ratio"].astype("Float64").round(2)
    metrics["Put_Vanna_Ratio"] = metrics["Put_Vanna_Ratio"].astype("Float64").round(2)
    metrics["Vanna_GEX_Total"] = metrics["Vanna_GEX_Total"].astype("Float64").round(2)

    metrics.insert(
        metrics.columns.get_loc("Vanna_GEX_Total") + 1,
        "DateTime",
        timestamp_str,
    )

    median_ivxoi = float(metrics["IVxOI"].median(skipna=True)) if not metrics["IVxOI"].dropna().empty else None
    metrics["Median_IVxOI"] = median_ivxoi
    metrics["Energy_Score"] = metrics["IVxOI"].apply(_score_energy, median_ivxoi=median_ivxoi)

    metrics["Regime"] = metrics.apply(_classify_regime, axis=1, iv_direction=direction_value)
    metrics["Dealer_Bias"] = metrics.apply(_dealer_bias, axis=1, iv_direction=direction_value)
    metrics["Bias"] = metrics.apply(
        lambda row: _bias_signal(row["Energy_Score"], row["Regime"]), axis=1
    )

    rel_spot = spot_price
    if rel_spot is None and "Spot" in unified_df.columns:
        try:
            rel_spot = float(pd.to_numeric(unified_df["Spot"], errors="coerce").dropna().iloc[0])
        except IndexError:
            rel_spot = None
    if rel_spot is not None and rel_spot > 0:
        metrics["Rel_Dist"] = ((metrics["Strike"] - rel_spot).abs() / rel_spot).round(4)
    else:
        metrics["Rel_Dist"] = pd.NA

    metrics["dGEX/dSpot"] = pd.NA
    metrics["dGEX/dSpot Rank"] = ""

    dgex_values: dict[float, float] = {}
    if rel_spot is not None and rel_spot > 0:
        bump_size = max(10.0, rel_spot * 0.0025)

        strikes = (
            pd.to_numeric(unified_df["Strike"], errors="coerce")
            .dropna()
            .unique()
        )

        sorted_strikes = sorted(strikes)

        if bump_size > 0 and len(sorted_strikes):
            contract_multiplier = 100.0
            safe_call_gamma = call_gamma.fillna(0)
            safe_put_gamma = put_gamma.fillna(0)
            safe_call_oi = call_open_interest.fillna(0)
            safe_put_oi = put_open_interest.fillna(0)

            closest_idx = min(
                range(len(sorted_strikes)),
                key=lambda idx: abs(sorted_strikes[idx] - rel_spot),
            )
            start_idx = max(0, closest_idx - 15)
            end_idx = min(len(sorted_strikes), closest_idx + 16)
            strikes_for_dgex = sorted_strikes[start_idx:end_idx]

            def _net_gex_for_spot(spot_level: float) -> float:
                spot_term = float(spot_level) ** 2 * contract_multiplier
                call_component = (safe_call_gamma * safe_call_oi * spot_term).sum()
                put_component = (safe_put_gamma * safe_put_oi * spot_term).sum()
                return float(call_component - put_component)

            for strike in strikes_for_dgex:
                strike_value = float(strike)
                gex_up = _net_gex_for_spot(strike_value + bump_size)
                gex_down = _net_gex_for_spot(strike_value - bump_size)
                derivative = (gex_up - gex_down) / (2 * bump_size)
                dgex_values[strike_value] = round(float(derivative), 2)
                strike_mask = metrics["Strike"] == strike_value
                metrics.loc[strike_mask, "dGEX/dSpot"] = dgex_values[strike_value]

            if dgex_values:
                dgex_series = pd.Series(dgex_values)
                top_n = min(5, len(dgex_series))
                top_indices = dgex_series.nlargest(top_n).index
                bottom_indices = dgex_series.nsmallest(top_n).index
                bottom_exclusive = [idx for idx in bottom_indices if idx not in set(top_indices)]

                for rank, strike_value in enumerate(top_indices, start=1):
                    strike_label = _format_strike(strike_value)
                    metrics.loc[
                        metrics["Strike"] == strike_value, "dGEX/dSpot Rank"
                    ] = f"Top {rank}: {strike_label}"

                for rank, strike_value in enumerate(bottom_exclusive, start=1):
                    strike_label = _format_strike(strike_value)
                    metrics.loc[
                        metrics["Strike"] == strike_value, "dGEX/dSpot Rank"
                    ] = f"Bottom {rank}: {strike_label}"

    metrics["Call_Vanna_Rank"] = ""
    metrics["Put_Vanna_Rank"] = ""
    metrics["Net_GEX_Highlight"] = ""
    metrics["DEX_highlight"] = ""
    metrics["Call_TEX_Highlight"] = ""
    metrics["Put_TEX_Highlight"] = ""
    metrics["TEX_highlight"] = ""
    metrics["Call_IVxOI_Rank"] = ""
    metrics["Put_IVxOI_Rank"] = ""

    metrics["Top5_Regime_Energy_Bias"] = ""

    if not metrics.empty:
        apply_highlight_annotations(metrics)

        activity_metric = pd.Series([0.0] * len(metrics), index=metrics.index, dtype="Float64")
        if "call_volume" in unified_df.columns or "puts_volume" in unified_df.columns:
            call_volume = pd.to_numeric(unified_df.get("call_volume", 0), errors="coerce").fillna(0)
            put_volume = pd.to_numeric(unified_df.get("puts_volume", 0), errors="coerce").fillna(0)
            activity_metric = (call_volume + put_volume).astype(float)
        if activity_metric.isna().all() or (activity_metric == 0).all():
            call_oi = pd.to_numeric(unified_df.get("call_open_interest", 0), errors="coerce").fillna(0)
            put_oi = pd.to_numeric(unified_df.get("puts_open_interest", 0), errors="coerce").fillna(0)
            activity_metric = (call_oi + put_oi).astype(float)

        top_n = min(5, len(metrics))
        if top_n:
            top_indices = activity_metric.nlargest(top_n).index
            for idx in top_indices:
                strike_value = _format_strike(metrics.at[idx, "Strike"])
                metrics.at[idx, "Top5_Regime_Energy_Bias"] = (
                    f"{strike_value}: {metrics.at[idx, 'Regime']} | "
                    f"{metrics.at[idx, 'Energy_Score']} | {metrics.at[idx, 'Dealer_Bias']}"
                )

    net_gex_above_spot: float | None = None
    net_gex_below_spot: float | None = None
    net_dex_above_spot: float | None = None
    net_dex_below_spot: float | None = None
    gamma_box_high: float | None = None
    gamma_box_low: float | None = None
    breakout_up: bool | None = None
    breakout_down: bool | None = None
    vex_dir_box_high: int | None = None
    vex_dir_box_low: int | None = None
    tex_dir_box_high: int | None = None
    tex_dir_box_low: int | None = None

    def _value_at_strike(series: pd.Series | None, strike: float | None) -> float | None:
        if series is None or strike is None:
            return None

        try:
            strike_value = float(strike)
        except (TypeError, ValueError):
            return None

        strike_series = pd.to_numeric(metrics.get("Strike"), errors="coerce")
        mask = strike_series == strike_value
        if not mask.any():
            return None

        values = pd.to_numeric(series.where(mask), errors="coerce").dropna()
        if values.empty:
            return None

        return float(values.iloc[0])

    if spot_price is not None and pd.notna(spot_price) and not metrics.empty:
        strike_values = pd.to_numeric(metrics.get("Strike"), errors="coerce")
        gex_series = pd.to_numeric(metrics.get("Net_GEX"), errors="coerce")
        dex_series = pd.to_numeric(metrics.get("Net_DEX"), errors="coerce")

        if not strike_values.isna().all():
            spot_level = float(spot_price)
            above_mask = strike_values > spot_level
            below_mask = strike_values < spot_level

            if "Net_GEX" in metrics.columns:
                net_gex_above_spot = float(gex_series.where(above_mask).sum())
                net_gex_below_spot = float(gex_series.where(below_mask).sum())

                positive_gex = gex_series > 0
                positive_above = strike_values.where(positive_gex & above_mask).dropna()
                positive_below = strike_values.where(positive_gex & below_mask).dropna()

                if not positive_above.empty:
                    gamma_box_high = float(positive_above.min())

                if not positive_below.empty:
                    gamma_box_low = float(positive_below.max())

            if "Net_DEX" in metrics.columns:
                net_dex_above_spot = float(dex_series.where(above_mask).sum())
                net_dex_below_spot = float(dex_series.where(below_mask).sum())

    metrics.attrs["total_net_DEX"] = float(metrics["Net_DEX"].fillna(0).sum())

    call_vex_total = None
    put_vex_total = None
    if "Call VEX" in metrics.columns and "Put VEX" in metrics.columns:
        call_vex_total = float(
            pd.to_numeric(metrics["Call VEX"], errors="coerce").fillna(0).sum()
        )
        put_vex_total = float(
            pd.to_numeric(metrics["Put VEX"], errors="coerce").fillna(0).sum()
        )
        metrics.attrs["vex_direction"] = _sign(call_vex_total + put_vex_total)

    if "Net_TEX" in metrics.columns:
        net_tex_total = float(
            pd.to_numeric(metrics["Net_TEX"], errors="coerce").fillna(0).sum()
        )
        metrics.attrs["total_net_TEX"] = net_tex_total
        metrics.attrs["tex_direction"] = _sign(net_tex_total)
    if "Net VEX" in metrics.columns:
        net_vex_total = float(
            pd.to_numeric(metrics["Net VEX"], errors="coerce").fillna(0).sum()
        )
        metrics.attrs["total_net_VEX"] = net_vex_total
    metrics.attrs["median_ivxoi"] = median_ivxoi
    metrics.attrs["iv_direction"] = direction_value

    if net_gex_above_spot is not None:
        metrics.attrs["net_gex_above_spot"] = net_gex_above_spot
    if net_gex_below_spot is not None:
        metrics.attrs["net_gex_below_spot"] = net_gex_below_spot
    if net_dex_above_spot is not None:
        metrics.attrs["net_dex_above_spot"] = net_dex_above_spot
    if net_dex_below_spot is not None:
        metrics.attrs["net_dex_below_spot"] = net_dex_below_spot
    if gamma_box_high is not None:
        metrics.attrs["gamma_box_high"] = gamma_box_high
        dgex_high = _value_at_strike(metrics.get("dGEX/dSpot"), gamma_box_high)
        breakout_up = dgex_high > 0 if dgex_high is not None else None
        if breakout_up is not None:
            metrics.attrs["gamma_box_breakout_up"] = breakout_up

        call_vex_high = _value_at_strike(metrics.get("Call VEX"), gamma_box_high)
        put_vex_high = _value_at_strike(metrics.get("Put VEX"), gamma_box_high)
        if call_vex_high is not None or put_vex_high is not None:
            vex_dir_box_high = _sign((call_vex_high or 0) + (put_vex_high or 0))
            metrics.attrs["vex_dir_box_high"] = vex_dir_box_high

        net_tex_high = _value_at_strike(metrics.get("Net_TEX"), gamma_box_high)
        if net_tex_high is not None:
            tex_dir_box_high = _sign(net_tex_high)
            metrics.attrs["tex_dir_box_high"] = tex_dir_box_high
    if gamma_box_low is not None:
        metrics.attrs["gamma_box_low"] = gamma_box_low
        dgex_low = _value_at_strike(metrics.get("dGEX/dSpot"), gamma_box_low)
        breakout_down = dgex_low < 0 if dgex_low is not None else None
        if breakout_down is not None:
            metrics.attrs["gamma_box_breakout_down"] = breakout_down

        call_vex_low = _value_at_strike(metrics.get("Call VEX"), gamma_box_low)
        put_vex_low = _value_at_strike(metrics.get("Put VEX"), gamma_box_low)
        if call_vex_low is not None or put_vex_low is not None:
            vex_dir_box_low = _sign((call_vex_low or 0) + (put_vex_low or 0))
            metrics.attrs["vex_dir_box_low"] = vex_dir_box_low

        net_tex_low = _value_at_strike(metrics.get("Net_TEX"), gamma_box_low)
        if net_tex_low is not None:
            tex_dir_box_low = _sign(net_tex_low)
            metrics.attrs["tex_dir_box_low"] = tex_dir_box_low

    market_state = classify_market_state(
        net_gex_above_spot,
        net_gex_below_spot,
        net_dex_above_spot,
        net_dex_below_spot,
    )
    metrics.attrs["market_state"] = market_state.scenario
    metrics.attrs["market_state_description"] = MARKET_STATE_DESCRIPTIONS.get(
        market_state.scenario or "",
        "",
    )
    playbook = MARKET_STATE_PLAYBOOK.get(market_state.scenario or "")
    if playbook:
        metrics.attrs["market_state_playbook"] = {
            "next_step": playbook.next_step,
            "useful_metrics": list(playbook.useful_metrics),
            "avoid": playbook.avoid,
        }
    metrics.attrs["market_state_components"] = {
        "GEX_location": market_state.gex_location,
        "GEX_sign": market_state.gex_sign,
        "DEX_location": market_state.dex_location,
        "DEX_sign": market_state.dex_sign,
        "GEX_zero": market_state.gex_zero,
        "DEX_zero": market_state.dex_zero,
        "Regime_Flip": market_state.regime_flip,
    }

    drop_set = {column for column in (drop_columns or []) if column in metrics.columns}
    if drop_set:
        metrics = metrics.drop(columns=sorted(drop_set))

    summary_column = "Summary"
    if summary_column not in metrics.columns:
        metrics = metrics.copy()
        metrics[summary_column] = ""

    header_order = (
        DERIVED_CSV_HEADER_TOP5_FIRST
        if len(metrics.index) >= 8
        else DERIVED_CSV_HEADER
    )

    column_order = (
        ["Strike"]
        + [
            column for column in header_order if column != "Strike" and column in metrics.columns
        ]
        if "Strike" in metrics.columns
        else [column for column in header_order if column in metrics.columns]
    )
    result = metrics.loc[:, column_order].copy()

    label_column = next((col for col in result.columns if col != "Strike"), None)

    has_totals = False
    if include_totals_row and not result.empty:
        base_rows = result.copy()
        label_target = label_column or "Strike"
        totals_row: dict[str, object] = {column: "" for column in result.columns}
        totals_row[label_target] = "Total"

        for column in result.columns:
            if column == "Strike" or column == label_target:
                continue
            if column in TOTAL_SUM_COLUMNS:
                numeric_series = pd.to_numeric(base_rows[column], errors="coerce")
                total_value = numeric_series.sum(skipna=True)
                if pd.isna(total_value):
                    totals_row[column] = ""
                else:
                    totals_row[column] = round(float(total_value), 2)
            else:
                totals_row[column] = ""
        result = pd.concat([result, pd.DataFrame([totals_row])], ignore_index=True)
        has_totals = True

        def _summary_row(label: str, column: str, value: float | None) -> dict[str, object]:
            row = {col: "" for col in result.columns}
            label_target = label_column or "Strike"
            row[label_target] = label
            if value is not None and column in row:
                row[column] = round(float(value), 2)
            return row

        summary_rows = []
        if net_gex_above_spot is not None:
            summary_rows.append(
                _summary_row("Net_GEX_Above_Spot", "Net_GEX", net_gex_above_spot)
            )
        if net_gex_below_spot is not None:
            summary_rows.append(
                _summary_row("Net_GEX_Below_Spot", "Net_GEX", net_gex_below_spot)
            )
        if net_dex_above_spot is not None:
            summary_rows.append(
                _summary_row("Net_DEX_Above_Spot", "Net_DEX", net_dex_above_spot)
            )
        if net_dex_below_spot is not None:
            summary_rows.append(
                _summary_row("Net_DEX_Below_Spot", "Net_DEX", net_dex_below_spot)
            )

        if append_market_state_row and market_state.scenario:
            description = metrics.attrs.get("market_state_description", "")
            state_summary = f"Market State: {market_state.scenario}"
            if description:
                state_summary += f" — {description}"

            state_row = {col: "" for col in result.columns}
            label_target = label_column or "Strike"
            state_row[label_target] = state_summary
            summary_rows.append(state_row)

        if summary_rows:
            result = pd.concat([result, pd.DataFrame(summary_rows)], ignore_index=True)

    result.attrs.update(metrics.attrs)
    result.attrs["has_totals_row"] = has_totals

    return result


__all__ = [
    "DERIVED_CSV_HEADER",
    "apply_highlight_annotations",
    "classify_market_state",
    "compute_derived_metrics",
    "MarketState",
]
