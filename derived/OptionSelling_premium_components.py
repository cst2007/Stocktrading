"""Utilities for computing premium-selling components from option data.

The calculations mirror a simple scoring approach that normalizes theta and open
interest values for calls and puts, then produces weighted components for
covered calls (CC) and cash-secured puts (CSP).
"""
from __future__ import annotations

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
        - "Call_Theta"
        - "Call_OI"
        - "Put_Theta"
        - "Put_OI"

    The resulting dataframe includes the strike and four weighted components:
    ``CC_Theta_Component``, ``CC_OI_Component``, ``CSP_Theta_Component``, and
    ``CSP_OI_Component``.
    """

    required_columns = {"Strike", "Call_Theta", "Call_OI", "Put_Theta", "Put_OI"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    df = df.copy()

    df["Call_Theta_norm_abs"] = _normalize_abs(df["Call_Theta"])
    df["Call_OI_norm"] = _normalize(df["Call_OI"])

    df["Put_Theta_norm_abs"] = _normalize_abs(df["Put_Theta"])
    df["Put_OI_norm"] = _normalize(df["Put_OI"])

    df["CC_Theta_Component"] = 0.25 * df["Call_Theta_norm_abs"]
    df["CC_OI_Component"] = 0.15 * df["Call_OI_norm"]

    df["CSP_Theta_Component"] = 0.25 * df["Put_Theta_norm_abs"]
    df["CSP_OI_Component"] = 0.15 * df["Put_OI_norm"]

    return df[[
        "Strike",
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
            "Call_Theta": [-12.5, -10.0, -8.0],
            "Call_OI": [15000, 18000, 13000],
            "Put_Theta": [-14.0, -11.0, -9.5],
            "Put_OI": [16000, 15500, 16500],
        }
    )

    components = build_premium_components(sample)
    print(components)
