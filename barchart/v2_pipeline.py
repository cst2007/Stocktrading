"""Phase 2 Barchart Options Analyzer pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

EPSILON = 1e-8


@dataclass(slots=True)
class ExposureRunConfig:
    ticker: str
    expiry: str
    spot: float
    contract_multiplier: float = 100.0
    delta_s: float = 10.0


@dataclass(slots=True)
class ExposureOutputs:
    core_path: Path
    side_path: Path
    reactivity_path: Path


# ---------------------------------------------------------------------------
# Input loading and normalization helpers
# ---------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=lambda col: str(col).strip().lower().replace(" ", "_"))
    renamed.columns = [col.replace("-", "_") for col in renamed.columns]
    return renamed


def _standardize_option_type(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().str[0]


def _ensure_required_columns(df: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} file is missing required columns: {', '.join(sorted(missing))}")


def load_options_file(path: Path) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))
    _ensure_required_columns(
        df,
        required=["ticker", "expiry_date", "option_type", "strike", "open_interest"],
        label="Options",
    )
    df["option_type"] = _standardize_option_type(df["option_type"])
    numeric_cols = ["strike", "open_interest"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["strike", "open_interest"])


def load_greeks_file(path: Path) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))
    # Support common IV naming variations
    if "iv" in df.columns and "implied_vol" not in df.columns:
        df.rename(columns={"iv": "implied_vol"}, inplace=True)
    if "implied_volatility" in df.columns and "implied_vol" not in df.columns:
        df.rename(columns={"implied_volatility": "implied_vol"}, inplace=True)

    _ensure_required_columns(
        df,
        required=["ticker", "expiry_date", "option_type", "strike", "implied_vol", "delta", "gamma", "theta"],
        label="Greeks",
    )
    df["option_type"] = _standardize_option_type(df["option_type"])
    numeric_cols = ["strike", "implied_vol", "delta", "gamma", "theta"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "vanna" in df.columns:
        df["vanna"] = pd.to_numeric(df["vanna"], errors="coerce")
    return df.dropna(subset=["strike"])


def combine_option_greeks(options_df: pd.DataFrame, greeks_df: pd.DataFrame) -> pd.DataFrame:
    merged = options_df.merge(
        greeks_df,
        on=["ticker", "expiry_date", "option_type", "strike"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No matching rows after joining options and greeks files")
    return merged


# ---------------------------------------------------------------------------
# Exposure calculations
# ---------------------------------------------------------------------------

def _pivot_per_strike(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["option_type"] = merged["option_type"].str.upper()

    metrics_map = {
        "open_interest": "OI",
        "delta": "Delta",
        "gamma": "Gamma",
        "implied_vol": "IV",
        "theta": "Theta",
        "vanna": "Vanna",
    }

    per_side: Dict[Tuple[float, str], Dict[str, float]] = {}
    for _, row in merged.iterrows():
        strike = float(row["strike"])
        opt_type = str(row["option_type"]).upper()
        key = (strike, opt_type)
        per_side[key] = per_side.get(key, {})
        for source, target in metrics_map.items():
            if source in row:
                per_side[key][target] = (
                    float(row.get(source, np.nan)) if pd.notna(row.get(source)) else np.nan
                )

    strikes = sorted({strike for strike, _ in per_side.keys()})
    per_strike = pd.DataFrame({"Strike": strikes})

    for strike, opt_type in per_side:
        prefix = "Call" if opt_type == "C" else "Put"
        for metric, value in per_side[(strike, opt_type)].items():
            column = f"{prefix}_{metric}"
            per_strike.loc[per_strike["Strike"] == strike, column] = value

    for column in per_strike.columns:
        if column != "Strike":
            per_strike[column] = pd.to_numeric(per_strike[column], errors="coerce").fillna(0.0)

    return per_strike.sort_values("Strike").reset_index(drop=True)


def _compute_exposures(per_strike: pd.DataFrame, config: ExposureRunConfig) -> pd.DataFrame:
    df = per_strike.copy()
    S = float(config.spot)
    M = float(config.contract_multiplier)

    df["Call_DEX"] = df["Call_Delta"] * S * df["Call_OI"] * M
    df["Put_DEX"] = df["Put_Delta"] * S * df["Put_OI"] * M
    df["Net_DEX"] = df["Call_DEX"] - df["Put_DEX"]

    df["Call_GEX"] = df["Call_Gamma"] * (S ** 2) * df["Call_OI"] * M
    df["Put_GEX"] = df["Put_Gamma"] * (S ** 2) * df["Put_OI"] * M
    df["Net_GEX"] = df["Call_GEX"] - df["Put_GEX"]
    df["Total_GEX"] = df["Call_GEX"] + df["Put_GEX"]

    df["Call_Vanna_Exp"] = df["Call_Vanna"] * df["Call_OI"] * M
    df["Put_Vanna_Exp"] = df["Put_Vanna"] * df["Put_OI"] * M
    df["Net_Vanna"] = df["Call_Vanna_Exp"] + df["Put_Vanna_Exp"]

    df["Call_Theta_Exp"] = df["Call_Theta"] * df["Call_OI"] * M
    df["Put_Theta_Exp"] = df["Put_Theta"] * df["Put_OI"] * M
    df["Net_Theta_Exp"] = df["Call_Theta_Exp"] + df["Put_Theta_Exp"]

    df["Call_IV_OI"] = df["Call_IV"] * df["Call_OI"]
    df["Put_IV_OI"] = df["Put_IV"] * df["Put_OI"]
    df["Net_IV_OI"] = df["Call_IV_OI"] + df["Put_IV_OI"]

    df["OI_Imbalance"] = df["Call_OI"] - df["Put_OI"]

    return df


def _net_gex_at_spot(exposures: pd.DataFrame, spot: float, contract_multiplier: float) -> float:
    call = exposures["Call_Gamma"] * (spot ** 2) * exposures["Call_OI"] * contract_multiplier
    put = exposures["Put_Gamma"] * (spot ** 2) * exposures["Put_OI"] * contract_multiplier
    return float((call - put).sum())


def _compute_marginal_gamma(df: pd.DataFrame, config: ExposureRunConfig) -> float:
    sup = config.spot + config.delta_s
    sdown = max(config.spot - config.delta_s, EPSILON)
    net_up = _net_gex_at_spot(df, sup, config.contract_multiplier)
    net_down = _net_gex_at_spot(df, sdown, config.contract_multiplier)
    return (net_up - net_down) / (2 * config.delta_s)


def _normalize(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    return (series - mean) / (std + EPSILON)


def _compute_scores(df: pd.DataFrame, d_gex_d_spot: float) -> pd.DataFrame:
    df = df.copy()
    df["dGEX_dSpot"] = float(d_gex_d_spot)

    df["Net_DEX_Rank"] = df["Net_DEX"].abs().rank(ascending=False, method="min").astype(int)
    df["Net_Theta_Rank"] = df["Net_Theta_Exp"].abs().rank(ascending=False, method="min").astype(int)
    df["Net_Vanna_Rank"] = df["Net_Vanna"].abs().rank(ascending=False, method="min").astype(int)

    to_normalize = {
        "Net_GEX": df["Net_GEX"],
        "dGEX_dSpot": df["dGEX_dSpot"],
        "Net_DEX": df["Net_DEX"],
        "Net_Theta_Exp": df["Net_Theta_Exp"],
        "Net_Vanna": df["Net_Vanna"],
    }
    normed: Dict[str, pd.Series] = {name: _normalize(series) for name, series in to_normalize.items()}

    abs_norm = {name: series.abs() for name, series in normed.items()}

    weights = {
        "Net_GEX": 0.30,
        "dGEX_dSpot": 0.25,
        "Net_DEX": 0.20,
        "Net_Theta_Exp": 0.15,
        "Net_Vanna": 0.10,
    }

    raw_score = sum(weights[name] * abs_norm[name] for name in weights)
    raw_min = raw_score.min()
    raw_max = raw_score.max()
    df["ReactivityScore_raw"] = raw_score
    df["ReactivityScore"] = 100 * (raw_score - raw_min) / ((raw_max - raw_min) + EPSILON)
    df["ReactivityRank"] = df["ReactivityScore"].rank(ascending=False, method="min").astype(int)

    try:
        threshold = np.nanpercentile(df["ReactivityScore"], 80)
    except IndexError:
        threshold = 0.0
    df["Is_High_Reactivity"] = df["ReactivityScore"] >= threshold

    df["Behavior_Tag"] = _assign_behavior_tags(df, abs_norm, weights)

    return df


def _assign_behavior_tags(df: pd.DataFrame, abs_norm: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    tags: List[str] = []
    for idx, row in df.iterrows():
        score = float(row.get("ReactivityScore", 0.0))
        if score < 20:
            tags.append("Low_Reactivity")
            continue

        net_gex = float(row.get("Net_GEX", 0.0))
        dgex = float(row.get("dGEX_dSpot", 0.0))
        gex_abs = float(abs_norm["Net_GEX"].get(idx, 0.0))
        dgex_abs = float(abs_norm["dGEX_dSpot"].get(idx, 0.0))
        dex_abs = float(abs_norm["Net_DEX"].get(idx, 0.0))
        theta_abs = float(abs_norm["Net_Theta_Exp"].get(idx, 0.0))
        vanna_abs = float(abs_norm["Net_Vanna"].get(idx, 0.0))

        contributions = {
            "gex": weights["Net_GEX"] * gex_abs,
            "dgex": weights["dGEX_dSpot"] * dgex_abs,
            "dex": weights["Net_DEX"] * dex_abs,
            "theta": weights["Net_Theta_Exp"] * theta_abs,
            "vanna": weights["Net_Vanna"] * vanna_abs,
        }
        total_contribution = sum(contributions.values()) + EPSILON
        dominant = max(contributions, key=contributions.get)
        dominant_share = contributions[dominant] / total_contribution

        if net_gex > 0 and dgex > 0 and gex_abs > 1 and dgex_abs > 1:
            tags.append("Fade_Zone")
        elif net_gex < 0 and dgex < 0 and gex_abs > 1 and dgex_abs > 1:
            tags.append("Continuation_Zone")
        elif dominant == "dex" and dominant_share >= 0.4:
            tags.append("Magnet_Zone")
        elif dominant == "vanna" and dominant_share >= 0.4:
            tags.append("Vol_Melt_Zone")
        elif dominant == "theta" and dominant_share >= 0.4:
            tags.append("Time_Decay_Trap")
        else:
            tags.append("Mixed_High_Reactivity")
    return pd.Series(tags, index=df.index)


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def run_exposure_pipeline(
    options_path: Path, greeks_path: Path, config: ExposureRunConfig, *, output_dir: Path
) -> ExposureOutputs:
    options_df = load_options_file(options_path)
    greeks_df = load_greeks_file(greeks_path)
    merged = combine_option_greeks(options_df, greeks_df)

    per_strike = _pivot_per_strike(merged)
    exposures = _compute_exposures(per_strike, config)
    d_gex_d_spot = _compute_marginal_gamma(per_strike, config)
    scored = _compute_scores(exposures, d_gex_d_spot)

    side_columns = [
        "Ticker",
        "Expiry",
        "Spot",
        "Strike",
        "Call_OI",
        "Put_OI",
        "OI_Imbalance",
        "Call_DEX",
        "Put_DEX",
        "Net_DEX",
        "Net_DEX_Rank",
        "Call_Theta_Exp",
        "Put_Theta_Exp",
        "Net_Theta_Exp",
        "Net_Theta_Rank",
        "Call_IV_OI",
        "Put_IV_OI",
        "Net_IV_OI",
        "Call_Vanna_Exp",
        "Put_Vanna_Exp",
        "Net_Vanna",
        "Net_Vanna_Rank",
        "Call_GEX",
        "Put_GEX",
        "Net_GEX",
        "Total_GEX",
        "dGEX_dSpot",
    ]

    reactivity_columns = [
        "Ticker",
        "Expiry",
        "Spot",
        "Strike",
        "Net_GEX",
        "dGEX_dSpot",
        "Net_DEX",
        "Net_Theta_Exp",
        "Net_Vanna",
        "ReactivityScore",
        "ReactivityRank",
        "Is_High_Reactivity",
        "Behavior_Tag",
        "Distance_to_Spot",
    ]

    core_columns = [
        "Ticker",
        "Expiry",
        "Spot",
        "Strike",
        "Call_OI",
        "Put_OI",
        "OI_Imbalance",
        "Net_GEX",
        "dGEX_dSpot",
        "Net_DEX",
        "Net_Theta_Exp",
        "Net_Vanna",
        "Behavior_Tag",
    ]

    scored.insert(0, "Ticker", config.ticker)
    scored.insert(1, "Expiry", config.expiry)
    scored["Spot"] = config.spot
    scored["Distance_to_Spot"] = scored["Strike"] - config.spot

    output_dir = output_dir.expanduser().resolve()
    core_dir = output_dir / "core"
    side_dir = output_dir / "details"
    signals_dir = output_dir / "signals"
    core_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)
    signals_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{config.ticker}-exp-{config.expiry}.csv"
    core_path = core_dir / f"CORE_EXPOSURES-{suffix}"
    side_path = side_dir / f"SIDE_EXPOSURES-{suffix}"
    reactivity_path = signals_dir / f"REACTIVITY_MAP-{suffix}"

    scored[core_columns].to_csv(core_path, index=False)
    scored[side_columns].to_csv(side_path, index=False)
    scored[reactivity_columns].to_csv(reactivity_path, index=False)

    return ExposureOutputs(core_path=core_path, side_path=side_path, reactivity_path=reactivity_path)


__all__ = [
    "ExposureRunConfig",
    "ExposureOutputs",
    "run_exposure_pipeline",
    "load_options_file",
    "load_greeks_file",
]
