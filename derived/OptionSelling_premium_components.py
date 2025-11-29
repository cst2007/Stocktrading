"""Utilities for computing premium-selling components from option data.

The calculations mirror a simple scoring approach that normalizes theta and open
interest values for calls and puts, then produces weighted components for
covered calls (CC) and cash-secured puts (CSP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON: float = 1e-9


def _normalize(series: pd.Series) -> pd.Series:
    """Return a z-score normalized series with a small epsilon for stability."""
    return (series - series.mean()) / (series.std() + EPSILON)


def _normalize_abs(series: pd.Series) -> pd.Series:
    """Return the absolute value of a normalized series."""
    return _normalize(series).abs()


def build_premium_components(df: pd.DataFrame) -> pd.DataFrame:
    """Generate weighted premium components for option-selling strategies.

    The input dataframe must contain the following columns:
        - "Strike"
        - "Net_GEX"
        - "Net_DEX"
        - "Call_Theta"
        - "Call_OI"
        - "Put_Theta"
        - "Put_OI"

    The resulting dataframe includes the strike, weighted components, and
    intermediate partial scores for each strategy. ``dGEX_Component`` uses a
    proxy slope of Net GEX across adjacent strikes to reward increasing values.
    """

    required_columns = {
        "Strike",
        "Net_GEX",
        "Net_DEX",
        "Call_Theta",
        "Call_OI",
        "Put_Theta",
        "Put_OI",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    df = df.copy()

    # Ensure strikes are processed in ascending order
    df = df.sort_values("Strike").reset_index(drop=True)

    df["Net_GEX_norm_abs"] = _normalize_abs(df["Net_GEX"])
    df["GEX_Component"] = 0.30 * df["Net_GEX_norm_abs"]

    gex_series = pd.to_numeric(df["Net_GEX"], errors="coerce").fillna(0.0)
    dex_series = pd.to_numeric(df["Net_DEX"], errors="coerce").fillna(0.0)

    gex_threshold = gex_series.abs().quantile(0.80)
    dex_threshold = dex_series.abs().quantile(0.80)
    magnet_mask = (gex_series.abs() > gex_threshold) | (dex_series.abs() > dex_threshold)

    gex_wall_threshold = gex_series.abs().quantile(0.85)
    dex_wall_threshold = dex_series.abs().quantile(0.85)
    wall_mask = (gex_series.diff().abs() > gex_wall_threshold) | (
        dex_series.diff().abs() > dex_wall_threshold
    )

    void_mask = (gex_series.abs() < gex_series.abs().quantile(0.30)) & (
        dex_series.abs() < dex_series.abs().quantile(0.30)
    )

    df["MWV"] = np.select(
        [magnet_mask, wall_mask, void_mask], ["Magnet", "Wall", "Void"], default=""
    )

    # Proxy for dGEX: reward positive slopes in Net_GEX across strikes
    df["Net_GEX_shift_up"] = df["Net_GEX"].shift(-1)
    df["Net_GEX_shift_down"] = df["Net_GEX"].shift(1)

    df["GEX_local_slope"] = df["Net_GEX_shift_up"] - df["Net_GEX_shift_down"]

    slope = df["GEX_local_slope"].fillna(0)
    df["GEX_slope_norm"] = (slope - slope.mean()) / (slope.std() + EPSILON)

    df["dGEX_Component"] = 0.20 * np.maximum(df["GEX_slope_norm"], 0)

    df.drop(["Net_GEX_shift_up", "Net_GEX_shift_down"], axis=1, inplace=True)

    df["Call_Theta_norm_abs"] = _normalize_abs(df["Call_Theta"])
    df["Call_OI_norm"] = _normalize(df["Call_OI"])

    df["Put_Theta_norm_abs"] = _normalize_abs(df["Put_Theta"])
    df["Put_OI_norm"] = _normalize(df["Put_OI"])

    df["CC_Theta_Component"] = 0.25 * df["Call_Theta_norm_abs"]
    df["CC_OI_Component"] = 0.15 * df["Call_OI_norm"]

    df["CSP_Theta_Component"] = 0.25 * df["Put_Theta_norm_abs"]
    df["CSP_OI_Component"] = 0.15 * df["Put_OI_norm"]

    df["CC_score_partial"] = (
        df["GEX_Component"]
        + df["dGEX_Component"]
        + df["CC_Theta_Component"]
        + df["CC_OI_Component"]
    )

    df["CSP_score_partial"] = (
        df["GEX_Component"]
        + df["dGEX_Component"]
        + df["CSP_Theta_Component"]
        + df["CSP_OI_Component"]
    )

    cc_ranks = df["CC_score_partial"].rank(method="min", ascending=False)
    csp_ranks = df["CSP_score_partial"].rank(method="min", ascending=False)

    df["CC_Rank"] = np.where(
        cc_ranks <= 7,
        "Rank " + cc_ranks.astype(int).astype(str) + ": " + df["Strike"].astype(str),
        "",
    )

    df["CSP_Rank"] = np.where(
        csp_ranks <= 7,
        "Rank " + csp_ranks.astype(int).astype(str) + ": " + df["Strike"].astype(str),
        "",
    )

    return df[[
        "Strike",
        "MWV",
        "CC_Rank",
        "CSP_Rank",
        "CC_score_partial",
        "CSP_score_partial",
        "GEX_Component",
        "dGEX_Component",
        "CC_Theta_Component",
        "CC_OI_Component",
        "CSP_Theta_Component",
        "CSP_OI_Component",
    ]]


if __name__ == "__main__":
    # Example usage for quick manual verification.
    sample = pd.DataFrame(
        {
            "Strike": [3950, 4000, 4050],
            "Net_GEX": [0.9, 1.2, 0.6],
            "Net_DEX": [1.1, -0.4, 0.2],
            "dGEX_dSpot": [0.05, 0.10, -0.02],
            "Call_Theta": [-12.5, -10.0, -8.0],
            "Call_OI": [15000, 18000, 13000],
            "Put_Theta": [-14.0, -11.0, -9.5],
            "Put_OI": [16000, 15500, 16500],
        }
    )

    components = build_premium_components(sample)
    print(components)
