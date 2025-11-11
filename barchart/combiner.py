"""Utilities for merging Barchart side-by-side and greeks CSV exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


# Column headers used when writing the combined CSV output. The order matches the
# column order returned by :func:`combine_option_files` so downstream tooling can
# rely on the human friendly labels produced by the CLI.
COMBINED_CSV_HEADER = [
    "Strike",
    "call_volume",
    "call_open_interest",
    "call_iv",
    "call_oi_iv",
    "call_delta",
    "call_gamma",
    "call_vega",
    "call_gex",
    "call_vanna",
    "puts_volume",
    "puts_open_interest",
    "puts_iv",
    "puts_oi_iv",
    "IVxOI",
    "puts_delta",
    "puts_gamma",
    "puts_vega",
    "puts_gex",
    "puts_vanna",
    "net_gex",
    "net_vanna",
    "Spot",
]


def _clean_strike(series: pd.Series) -> pd.Series:
    """Return a numeric strike column regardless of thousands separators."""

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _load_side_by_side(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = {
        "Type",
        "Volume",
        "Open Int",
        "IV",
        "Strike",
        "Type.1",
        "Volume.1",
        "Open Int.1",
        "IV.1",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "Side-by-side CSV is missing expected columns: " + ", ".join(sorted(missing))
        )

    def _clean_numeric(series: pd.Series, *, is_percent: bool = False) -> pd.Series:
        cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
        if is_percent:
            cleaned = cleaned.str.replace("%", "", regex=False)
            return pd.to_numeric(cleaned, errors="coerce") / 100.0
        return pd.to_numeric(cleaned, errors="coerce")

    side = pd.DataFrame(
        {
            "Strike": _clean_strike(df["Strike"]),
            "call_volume": _clean_numeric(df["Volume"]),
            "call_open_interest": _clean_numeric(df["Open Int"]),
            "call_iv": _clean_numeric(df["IV"], is_percent=True),
            "puts_volume": _clean_numeric(df["Volume.1"]),
            "puts_open_interest": _clean_numeric(df["Open Int.1"]),
            "puts_iv": _clean_numeric(df["IV.1"], is_percent=True),
        }
    )

    for column in (
        "call_volume",
        "call_open_interest",
        "puts_volume",
        "puts_open_interest",
    ):
        side[column] = side[column].round().astype("Int64")

    return side.dropna(subset=["Strike"]).reset_index(drop=True)


def _load_greeks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {
        "Strike",
        "Delta",
        "Gamma",
        "Vega",
        "Delta.1",
        "Gamma.1",
        "Vega.1",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "Greeks CSV is missing expected columns: " + ", ".join(sorted(missing))
        )

    greeks = pd.DataFrame(
        {
            "Strike": _clean_strike(df["Strike"]),
            "call_delta": pd.to_numeric(df["Delta"], errors="coerce"),
            "call_gamma": pd.to_numeric(df["Gamma"], errors="coerce"),
            "call_vega": pd.to_numeric(df["Vega"], errors="coerce"),
            "puts_delta": pd.to_numeric(df["Delta.1"], errors="coerce"),
            "puts_gamma": pd.to_numeric(df["Gamma.1"], errors="coerce"),
            "puts_vega": pd.to_numeric(df["Vega.1"], errors="coerce"),
        }
    )

    return greeks.dropna(subset=["Strike"]).reset_index(drop=True)


def combine_option_files(
    side_by_side_path: Path,
    greeks_path: Path,
    *,
    spot_price: float,
    contract_multiplier: float = 100.0,
) -> pd.DataFrame:
    """Merge side-by-side and greeks datasets into a single per-strike table."""

    side = _load_side_by_side(side_by_side_path)
    greeks = _load_greeks(greeks_path)

    merged = pd.merge(side, greeks, on="Strike", how="inner", validate="one_to_one")

    # Derived exposures follow contract specifications outlined in Phase 1.
    merged["call_gex"] = (
        merged["call_open_interest"] * merged["call_gamma"] * contract_multiplier
    )
    merged["puts_gex"] = (
        merged["puts_open_interest"] * merged["puts_gamma"] * contract_multiplier
    )
    merged["call_vanna"] = (
        merged["call_open_interest"] * (1 - merged["call_delta"]) * merged["call_vega"] * 100
    )
    merged["puts_vanna"] = (
        merged["puts_open_interest"]
        * (1 - merged["puts_delta"])
        * merged["puts_vega"]
        * 100
    )

    merged["call_oi_iv"] = merged["call_open_interest"] * merged["call_iv"]
    merged["puts_oi_iv"] = merged["puts_open_interest"] * merged["puts_iv"]

    merged["IVxOI"] = merged[["call_oi_iv", "puts_oi_iv"]].fillna(0).sum(axis=1)

    merged["net_gex"] = merged["call_gex"] + merged["puts_gex"]
    merged["net_vanna"] = merged["call_vanna"] + merged["puts_vanna"]

    merged["Spot"] = float(spot_price)

    rounding_map = {
        "call_iv": 1,
        "call_oi_iv": 1,
        "call_delta": 1,
        "call_gamma": 1,
        "call_vega": 1,
        "call_gex": 1,
        "call_vanna": 1,
        "IVxOI": 1,
        "puts_gex": 1,
        "puts_vanna": 1,
        "puts_iv": 1,
        "puts_oi_iv": 1,
        "puts_delta": 1,
        "puts_gamma": 1,
        "puts_vega": 1,
        "net_gex": 1,
        "net_vanna": 1,
        "Spot": 2,
    }
    for column, decimals in rounding_map.items():
        merged[column] = merged[column].round(decimals)

    columns = [
        "Strike",
        "call_volume",
        "call_open_interest",
        "call_iv",
        "call_oi_iv",
        "call_delta",
        "call_gamma",
        "call_vega",
        "call_gex",
        "call_vanna",
        "puts_volume",
        "puts_open_interest",
        "puts_iv",
        "puts_oi_iv",
        "IVxOI",
        "puts_delta",
        "puts_gamma",
        "puts_vega",
        "puts_gex",
        "puts_vanna",
        "net_gex",
        "net_vanna",
        "Spot",
    ]

    return merged[columns].sort_values("Strike").reset_index(drop=True)


def _write_output(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, header=COMBINED_CSV_HEADER)


def run_cli(args: Iterable[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Combine Barchart side-by-side and greeks CSV exports into a unified table.",
    )
    parser.add_argument("--side-by-side", required=True, help="Path to the side-by-side CSV file.")
    parser.add_argument("--greeks", required=True, help="Path to the volatility/greeks CSV file.")
    parser.add_argument(
        "--spot-price",
        required=True,
        type=float,
        help="Underlying spot price used for exposure calculations.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV file for the combined output.",
    )
    parser.add_argument(
        "--contract-multiplier",
        type=float,
        default=100.0,
        help="Contract size used in exposure calculations (default: 100).",
    )

    parsed = parser.parse_args(args=args)

    combined = combine_option_files(
        Path(parsed.side_by_side),
        Path(parsed.greeks),
        spot_price=parsed.spot_price,
        contract_multiplier=parsed.contract_multiplier,
    )

    output_path = Path(parsed.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_output(combined, output_path)
    return output_path


def main() -> None:  # pragma: no cover - CLI entry point
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
