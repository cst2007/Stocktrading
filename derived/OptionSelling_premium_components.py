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
        - "dGEX_dSpot" (or "dGEX/dSpot")
        - "Call_Theta"
        - "Call_OI"
        - "Put_Theta"
        - "Put_OI"

    The resulting dataframe includes the strike, weighted components, and
    intermediate partial scores for each strategy:
    ``GEX_Component``, ``dGEX_Component``, ``CC_Theta_Component``,
    ``CC_OI_Component``, ``CC_score_partial``, ``CSP_Theta_Component``,
    ``CSP_OI_Component``, and ``CSP_score_partial``.
    """

    required_columns = {"Strike", "Net_GEX", "Call_Theta", "Call_OI", "Put_Theta", "Put_OI"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    if "dGEX_dSpot" in df.columns:
        dgex_dspot_col = "dGEX_dSpot"
    elif "dGEX/dSpot" in df.columns:
        dgex_dspot_col = "dGEX/dSpot"
    else:
        raise ValueError(
            "Input dataframe is missing required column: dGEX_dSpot (or dGEX/dSpot)"
        )

    df = df.copy()

    # Clean and convert dGEX/dSpot values that may arrive as strings with commas
    df["dGEX_dSpot_num"] = (
        df[dgex_dspot_col].astype(str).str.replace(",", "", regex=False).astype(float)
    )

    df["Net_GEX_norm_abs"] = _normalize_abs(df["Net_GEX"])
    df["GEX_Component"] = 0.30 * df["Net_GEX_norm_abs"]

    dgex_series = df["dGEX_dSpot_num"].astype(float)
    dgex_min = dgex_series.min()
    dgex_max = dgex_series.max()
    dgex_range = dgex_max - dgex_min

    if abs(dgex_range) < EPSILON:
        df["dGEX_dSpot_norm"] = 0.0
        df["dGEX_Component"] = 0.0
    else:
        df["dGEX_dSpot_norm"] = (dgex_series - dgex_min) / (dgex_range + EPSILON)
        df["dGEX_Component"] = (
            0.20 * np.maximum(df["dGEX_dSpot_norm"], 0.0)
        ).astype(float)

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

    return df[[
        "Strike",
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
            "dGEX_dSpot": [0.05, 0.10, -0.02],
            "Call_Theta": [-12.5, -10.0, -8.0],
            "Call_OI": [15000, 18000, 13000],
            "Put_Theta": [-14.0, -11.0, -9.5],
            "Put_OI": [16000, 15500, 16500],
        }
    )

    components = build_premium_components(sample)
    print(components)
