"""Utilities for computing derived exposure metrics from unified datasets."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

DERIVED_CSV_HEADER = [
    "Strike",
    "DateTime",
    "call_Vanna",
    "puts_Vanna",
    "net_Vanna",
    "call_GEX",
    "puts_GEX",
    "net_GEX",
    "Vanna_GEX_Ratio",
    "Vanna_GEX_Call_Ratio",
]


def _format_timestamp(timestamp: datetime) -> str:
    utc_time = timestamp.astimezone(timezone.utc)
    return utc_time.replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def compute_derived_metrics(
    unified_df: pd.DataFrame, *, calculation_time: datetime
) -> pd.DataFrame:
    """Return the Phase 1 derived exposure metrics for ``unified_df``."""

    if "Strike" not in unified_df.columns:
        raise KeyError("Unified dataset must contain a 'Strike' column")

    timestamp_str = _format_timestamp(calculation_time)

    metrics = pd.DataFrame(
        {
            "Strike": unified_df["Strike"].astype(float),
            "DateTime": timestamp_str,
            "call_Vanna": unified_df["call_vanna"].astype(float),
            "puts_Vanna": unified_df["puts_vanna"].astype(float),
            "net_Vanna": unified_df["net_vanna"].astype(float),
            "call_GEX": unified_df["call_gex"].astype(float),
            "puts_GEX": unified_df["puts_gex"].astype(float),
            "net_GEX": unified_df["net_gex"].astype(float),
        }
    )

    net_gex_denom = metrics["net_GEX"].replace({0: pd.NA})
    call_gex_denom = metrics["call_GEX"].replace({0: pd.NA})

    metrics["Vanna_GEX_Ratio"] = metrics["net_Vanna"] / net_gex_denom
    metrics["Vanna_GEX_Call_Ratio"] = metrics["call_Vanna"] / call_gex_denom

    return metrics[DERIVED_CSV_HEADER]


__all__ = ["DERIVED_CSV_HEADER", "compute_derived_metrics"]
