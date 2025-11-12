"""Utilities for computing derived exposure metrics from unified datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Collection, Sequence

import pandas as pd

DERIVED_CSV_HEADER = [
    "Strike",
    "Call_Vanna",
    "Call_Vanna_Highlight",
    "Put_Vanna",
    "Put_Vanna_Highlight",
    "Net_Vanna",
    "Call_GEX",
    "Put_GEX",
    "Net_GEX",
    "Net_GEX_Highlight",
    "Call_DEX",
    "Put_DEX",
    "Net_DEX",
    "DEX_highlight",
    "Call_TEX",
    "Call_TEX_Highlight",
    "Put_TEX",
    "Put_TEX_Highlight",
    "Net_TEX",
    "TEX_highlight",
    "Call_IV",
    "Put_IV",
    "Call_IVxOI",
    "Put_IVxOI",
    "Call_IVxOI_Highlight",
    "Put_IVxOI_Highlight",
    "IVxOI",
    "Median_IVxOI",
    "Call_Vanna_Ratio",
    "Put_Vanna_Ratio",
    "Vanna_GEX_Total",
    "DateTime",
    "Energy_Score",
    "Regime",
    "Dealer_Bias",
    "IV_Direction",
    "Rel_Dist",
    "Top5_Regime_Energy_Bias",
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
    "Call_TEX",
    "Put_TEX",
    "Net_TEX",
    "Call_IVxOI",
    "Put_IVxOI",
    "IVxOI",
}


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


def apply_highlight_annotations(metrics: pd.DataFrame) -> None:
    """Populate highlight columns for the most notable strike metrics."""

    highlight_columns = [
        "Call_Vanna_Highlight",
        "Put_Vanna_Highlight",
        "Net_GEX_Highlight",
        "DEX_highlight",
        "Call_TEX_Highlight",
        "Put_TEX_Highlight",
        "TEX_highlight",
        "Call_IVxOI_Highlight",
        "Put_IVxOI_Highlight",
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

    _set_ranked_highlights("Call_Vanna", "Call_Vanna_Highlight", top_n=4)
    _set_ranked_highlights(
        "Put_Vanna",
        "Put_Vanna_Highlight",
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
    _set_ranked_highlights("Call_IVxOI", "Call_IVxOI_Highlight", top_n=4)
    _set_ranked_highlights("Put_IVxOI", "Put_IVxOI_Highlight", top_n=4)


def compute_derived_metrics(
    unified_df: pd.DataFrame,
    *,
    calculation_time: datetime,
    spot_price: float | None = None,
    iv_direction: str = "down",
    drop_columns: Sequence[str] | None = None,
    include_totals_row: bool = False,
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
                * unified_df["call_open_interest"].astype(float)
                * 100
            ),
            "Put_DEX": (
                unified_df["puts_delta"].astype(float)
                * unified_df["puts_open_interest"].astype(float)
                * 100
            ),
            "Call_TEX": (
                unified_df["call_theta"].astype(float)
                * unified_df["call_open_interest"].astype(float)
                * 100
            ),
            "Put_TEX": (
                unified_df["puts_theta"].astype(float)
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
    metrics["Put_Vanna_Highlight"] = ""
    metrics["Net_GEX_Highlight"] = ""
    metrics["DEX_highlight"] = ""
    metrics["Call_TEX_Highlight"] = ""
    metrics["Put_TEX_Highlight"] = ""
    metrics["TEX_highlight"] = ""
    metrics["Call_IVxOI_Highlight"] = ""
    metrics["Put_IVxOI_Highlight"] = ""

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

    metrics.attrs["total_net_DEX"] = float(metrics["Net_DEX"].fillna(0).sum())
    metrics.attrs["median_ivxoi"] = median_ivxoi
    metrics.attrs["iv_direction"] = direction_value

    drop_set = {column for column in (drop_columns or []) if column in metrics.columns}
    if drop_set:
        metrics = metrics.drop(columns=sorted(drop_set))

    column_order = [column for column in DERIVED_CSV_HEADER if column in metrics.columns]
    result = metrics.loc[:, column_order].copy()

    has_totals = False
    if include_totals_row and not result.empty:
        base_rows = result.copy()
        totals_row: dict[str, object] = {}
        for column in result.columns:
            if column == "Strike":
                totals_row[column] = "Total"
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

    result.attrs.update(metrics.attrs)
    result.attrs["has_totals_row"] = has_totals

    return result


__all__ = ["DERIVED_CSV_HEADER", "apply_highlight_annotations", "compute_derived_metrics"]
