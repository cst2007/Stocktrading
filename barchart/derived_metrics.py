"""Utilities for computing derived exposure metrics from unified datasets."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

DERIVED_CSV_HEADER = [
    "Strike",
    "DateTime",
    "Call_Vanna",
    "Put_Vanna",
    "Net_Vanna",
    "Call_GEX",
    "Put_GEX",
    "Net_GEX",
    "Call_DEX",
    "Put_DEX",
    "Net_DEX",
    "Call_IV",
    "Put_IV",
    "Call_IVxOI",
    "Put_IVxOI",
    "IVxOI",
    "Median_IVxOI",
    "Call_Vanna_Ratio",
    "Put_Vanna_Ratio",
    "Vanna_GEX_Total",
    "Energy_Score",
    "Regime",
    "Dealer_Bias",
    "IV_Direction",
    "Rel_Dist",
    "Top5_Regime_Energy_Bias",
    "Call_Vanna_Highlight",
    "Net_GEX_Highlight",
]


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


def compute_derived_metrics(
    unified_df: pd.DataFrame,
    *,
    calculation_time: datetime,
    spot_price: float | None = None,
    iv_direction: str = "down",
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
        "call_open_interest",
        "puts_open_interest",
        "call_delta",
        "puts_delta",
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

    metrics = pd.DataFrame(
        {
            "Strike": unified_df["Strike"].astype(float),
            "DateTime": timestamp_str,
            "Call_Vanna": unified_df["call_vanna"].astype(float),
            "Put_Vanna": unified_df["puts_vanna"].astype(float),
            "Net_Vanna": unified_df["net_vanna"].astype(float),
            "Call_GEX": unified_df["call_gex"].astype(float),
            "Put_GEX": unified_df["puts_gex"].astype(float),
            "Net_GEX": unified_df["net_gex"].astype(float),
            "Call_DEX": (
                unified_df["call_delta"].astype(float)
                * unified_df["call_open_interest"].astype(float)
                * 100
            ),
            "Put_DEX": (
                unified_df["puts_delta"].astype(float)
                * unified_df["puts_open_interest"].astype(float)
                * 100
            ),
            "Call_IV": unified_df["call_iv"].astype(float).round(1),
            "Put_IV": unified_df["puts_iv"].astype(float).round(1),
            "Call_IVxOI": unified_df["call_oi_iv"].astype(float).round(1),
            "Put_IVxOI": unified_df["puts_oi_iv"].astype(float).round(1),
        }
    )

    metrics["Net_DEX"] = metrics["Call_DEX"] + metrics["Put_DEX"]

    for column in ("Call_DEX", "Put_DEX", "Net_DEX"):
        metrics[column] = metrics[column].round(1)

    metrics["IVxOI"] = (
        metrics[["Call_IVxOI", "Put_IVxOI"]].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    )

    call_gex_denom = metrics["Call_GEX"].replace({0: pd.NA})
    put_gex_denom = metrics["Put_GEX"].replace({0: pd.NA})

    metrics["Call_Vanna_Ratio"] = metrics["Call_Vanna"] / call_gex_denom
    metrics["Put_Vanna_Ratio"] = metrics["Put_Vanna"] / put_gex_denom
    total_gex_denom = (metrics["Call_GEX"] + metrics["Put_GEX"]).replace({0: pd.NA})
    metrics["Vanna_GEX_Total"] = metrics["Net_Vanna"] / total_gex_denom

    metrics["Call_Vanna_Ratio"] = metrics["Call_Vanna_Ratio"].astype("Float64").round(2)
    metrics["Put_Vanna_Ratio"] = metrics["Put_Vanna_Ratio"].astype("Float64").round(2)
    metrics["Vanna_GEX_Total"] = metrics["Vanna_GEX_Total"].astype("Float64").round(2)

    median_ivxoi = float(metrics["IVxOI"].median(skipna=True)) if not metrics["IVxOI"].dropna().empty else None
    metrics["Median_IVxOI"] = median_ivxoi
    metrics["Energy_Score"] = metrics["IVxOI"].apply(_score_energy, median_ivxoi=median_ivxoi)

    metrics["IV_Direction"] = direction_value
    metrics["Regime"] = metrics.apply(_classify_regime, axis=1, iv_direction=direction_value)
    metrics["Dealer_Bias"] = metrics.apply(_dealer_bias, axis=1, iv_direction=direction_value)

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

    metrics["Call_Vanna_Highlight"] = ""
    metrics["Net_GEX_Highlight"] = ""

    metrics["Top5_Regime_Energy_Bias"] = ""

    if not metrics.empty:
        call_count = int(metrics["Call_Vanna"].count())
        if call_count:
            call_top = metrics["Call_Vanna"].nlargest(min(3, call_count)).index
            metrics.loc[call_top, "Call_Vanna_Highlight"] = "highlight"

        net_count = int(metrics["Net_GEX"].count())
        if net_count:
            net_top = metrics["Net_GEX"].nlargest(min(3, net_count)).index
            metrics.loc[net_top, "Net_GEX_Highlight"] = "highlight"

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

    metrics.attrs["total_net_DEX"] = float(metrics["Net_DEX"].fillna(0).sum())
    metrics.attrs["median_ivxoi"] = median_ivxoi
    metrics.attrs["iv_direction"] = direction_value

    return metrics[DERIVED_CSV_HEADER]


__all__ = ["DERIVED_CSV_HEADER", "compute_derived_metrics"]
