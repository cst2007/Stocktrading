"""Core analytics engine for the Barchart Options Analyzer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:  # pragma: no cover - optional dependency guard
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled at runtime
    plt = None

from .database import AnalyticsDatabase
from .derived_metrics import apply_highlight_annotations

logger = logging.getLogger(__name__)

HIGHLIGHT_METRIC_MAP = {
    "Net_DEX": "DEX_highlight",
    "Call_Vanna": "Call_Vanna_Highlight",
    "Put_Vanna": "Put_Vanna_Highlight",
    "Net_GEX": "Net_GEX_Highlight",
    "Call_TEX": "Call_TEX_Highlight",
    "Put_TEX": "Put_TEX_Highlight",
    "Net_TEX": "TEX_highlight",
    "Call_IVxOI": "Call_IVxOI_Highlight",
    "Put_IVxOI": "Put_IVxOI_Highlight",
}


def _normalize_iv_direction(value: str | None) -> str:
    direction = (value or "").strip().lower()
    if direction not in {"up", "down", "unknown"}:
        raise ValueError("iv_direction must be 'up', 'down', or 'unknown'")
    return direction


def _classify_regime(call_ratio: float | None, put_ratio: float | None, iv_direction: str) -> str:
    def _lt(val: float | None, threshold: float) -> bool:
        return val is not None and not pd.isna(val) and val < threshold

    def _gt(val: float | None, threshold: float) -> bool:
        return val is not None and not pd.isna(val) and val > threshold

    if _lt(call_ratio, 1) and _lt(put_ratio, 1):
        return "Gamma Pin"
    if _gt(call_ratio, 2) and iv_direction == "up":
        return "Pre-Earnings Fade"
    if _gt(put_ratio, 2) and iv_direction == "down":
        return "Post-Earnings Vanna Rally"
    if _gt(call_ratio, 2) and _lt(put_ratio, 1):
        return "Vol Drift Down"
    if _lt(call_ratio, 1) and _gt(put_ratio, 2):
        return "Vol Drift Up"
    return "Transition Zone"


def _score_energy(ivxoi: float | None, median_ivxoi: float | None) -> str:
    if ivxoi is None or pd.isna(ivxoi) or median_ivxoi is None or pd.isna(median_ivxoi):
        return "Low"
    if median_ivxoi <= 0:
        return "High" if ivxoi > 0 else "Low"
    if ivxoi > 1.5 * median_ivxoi:
        return "High"
    if ivxoi > 0.8 * median_ivxoi:
        return "Moderate"
    return "Low"


def _dealer_bias(call_ratio: float | None, put_ratio: float | None, iv_direction: str) -> str:
    call_gt = call_ratio is not None and not pd.isna(call_ratio) and call_ratio > 2
    put_gt = put_ratio is not None and not pd.isna(put_ratio) and put_ratio > 2
    if put_gt and iv_direction == "down":
        return "Dealer Buying → Bullish Drift"
    if call_gt and iv_direction == "down":
        return "Dealer Selling → Bearish Fade"
    return "Neutral / Mean Reversion"


@dataclass
class ProcessingResult:
    """Container for the metrics generated from a Barchart CSV."""

    ticker: str
    expiry: str
    total_gex: float
    total_vanna: float
    call_gex: float
    put_gex: float
    call_vanna: float
    put_vanna: float
    gex_by_strike: Dict[float, float]
    vanna_by_strike: Dict[float, float]
    source_path: Path
    output_directory: Path
    strike_type_metrics: List["StrikeTypeMetric"]
    call_vanna_ratio_by_strike: Dict[float, float | None]
    put_vanna_ratio_by_strike: Dict[float, float | None]
    vanna_gex_total_by_strike: Dict[float, float | None]
    energy_score_by_strike: Dict[float, str]
    regime_by_strike: Dict[float, str]
    dealer_bias_by_strike: Dict[float, str]
    ivxoi_by_strike: Dict[float, float | None]
    rel_dist_by_strike: Dict[float, float | None]
    top5_bias_summary: Dict[float, str]
    median_ivxoi: float | None
    iv_direction: str
    strike_summary_df: pd.DataFrame


@dataclass
class StrikeTypeMetric:
    strike: float
    option_type: str
    next_gex: float
    vanna: float
    iv: float | None


class MissingOpenInterestError(ValueError):
    """Raised when a CSV lacks open interest data required for GEX."""


class MissingGreeksError(ValueError):
    """Raised when a CSV does not include the Greeks required for analytics."""


class BarchartOptionsAnalyzer:
    """Analyze Barchart-formatted options chains.

    Parameters
    ----------
    contract_multiplier:
        Contract size used for exposure calculations (defaults to 100).
    create_charts:
        When True, strike-level PNG charts are produced in the output folder.
    """

    _FILENAME_PATTERN = re.compile(
        r"(?P<ticker>\$?[a-zA-Z0-9]+)-(?:options|volatility-greeks)-exp-(?P<expiry>\d{4}-\d{2}-\d{2}).*\.csv$"
    )

    NUMERIC_COLUMNS = {
        "strike",
        "delta",
        "gamma",
        "vega",
        "theta",
        "open_interest",
        "volume",
        "iv",
        "underlying_price",
        "mid",
        "bid",
        "ask",
        "gex",
        "vanna",
    }

    def __init__(
        self,
        contract_multiplier: float = 100.0,
        create_charts: bool = True,
        spot_price: float | None = None,
        iv_direction: str = "down",
    ) -> None:
        self.contract_multiplier = contract_multiplier
        self.create_charts = create_charts
        self.spot_price = spot_price
        self.iv_direction = _normalize_iv_direction(iv_direction)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_path(self, path: Path, output_directory: Path) -> List[ProcessingResult]:
        """Process a single CSV or every CSV within a directory."""

        path = path.expanduser().resolve()
        output_directory = output_directory.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        if path.is_file():
            try:
                return [self._process_file(path, output_directory)]
            except (MissingOpenInterestError, MissingGreeksError) as exc:
                logger.error("Skipping %s: %s", path, exc)
                return []

        if not path.exists():
            raise FileNotFoundError(f"Input path {path} does not exist")

        results: List[ProcessingResult] = []
        csv_paths = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.name.lower().endswith(".csv")
        )
        for csv_path in csv_paths:
            try:
                results.append(self._process_file(csv_path, output_directory))
            except (MissingOpenInterestError, MissingGreeksError) as exc:
                logger.error("Skipping %s: %s", csv_path, exc)
            except Exception:  # pragma: no cover - surfaced via CLI logging
                logger.exception("Failed to process %s", csv_path)
        if not results:
            logger.warning("No CSV files found under %s", path)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_file(self, path: Path, output_directory: Path) -> ProcessingResult:
        logger.info("Processing file: %s", path)
        ticker, expiry = self._parse_metadata_from_filename(path.name)
        df = self._load_csv(path)
        df = self._preprocess(df)
        metrics = self._compute_metrics(df)

        summary_df: pd.DataFrame = metrics["summary_by_strike"].copy()

        def _float_dict(series: pd.Series) -> Dict[float, float | None]:
            output: Dict[float, float | None] = {}
            for idx, value in series.items():
                key = float(idx)
                output[key] = None if pd.isna(value) else float(value)
            return output

        def _string_dict(series: pd.Series) -> Dict[float, str]:
            output: Dict[float, str] = {}
            for idx, value in series.items():
                if isinstance(value, str) and value:
                    output[float(idx)] = value
            return output

        result = ProcessingResult(
            ticker=ticker,
            expiry=expiry,
            total_gex=metrics["total_gex"],
            total_vanna=metrics["total_vanna"],
            call_gex=metrics["call_gex"],
            put_gex=metrics["put_gex"],
            call_vanna=metrics["call_vanna"],
            put_vanna=metrics["put_vanna"],
            gex_by_strike=metrics["gex_by_strike"],
            vanna_by_strike=metrics["vanna_by_strike"],
            source_path=path,
            output_directory=output_directory,
            strike_type_metrics=metrics["strike_type_metrics"],
            call_vanna_ratio_by_strike=_float_dict(summary_df["Call_Vanna_Ratio"]),
            put_vanna_ratio_by_strike=_float_dict(summary_df["Put_Vanna_Ratio"]),
            vanna_gex_total_by_strike=_float_dict(summary_df["Vanna_GEX_Total"]),
            energy_score_by_strike={float(idx): str(value) for idx, value in summary_df["Energy_Score"].items()},
            regime_by_strike={float(idx): str(value) for idx, value in summary_df["Regime"].items()},
            dealer_bias_by_strike={float(idx): str(value) for idx, value in summary_df["Dealer_Bias"].items()},
            ivxoi_by_strike=_float_dict(summary_df["IVxOI"]),
            rel_dist_by_strike=_float_dict(summary_df["Rel_Dist"]),
            top5_bias_summary=_string_dict(summary_df["Top5_Regime_Energy_Bias"]),
            median_ivxoi=metrics["median_ivxoi"],
            iv_direction=metrics["iv_direction"],
            strike_summary_df=summary_df,
        )

        self._write_outputs(result)
        return result

    def _parse_metadata_from_filename(self, filename: str) -> (str, str):
        unified_match = re.match(
            r"unified_(?P<ticker>[a-zA-Z0-9$]+)_(?P<expiry>\d{4}-\d{2}-\d{2})",
            filename,
            re.IGNORECASE,
        )
        if unified_match:
            return unified_match.group("ticker").upper(), unified_match.group("expiry")

        match = self._FILENAME_PATTERN.search(filename)
        if not match:
            logger.warning(
                "Filename %s does not follow expected convention; ticker/expiry set to 'UNKNOWN'.",
                filename,
            )
            return "UNKNOWN", "UNKNOWN"
        return match.group("ticker").upper(), match.group("expiry")

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # Trim whitespace around headers to make downstream matching resilient
        df.rename(columns=lambda column: column.strip(), inplace=True)
        df = self._reshape_if_side_by_side(df)
        logger.debug("Loaded %d rows", len(df))
        return df

    def _reshape_if_side_by_side(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert side-by-side call/put columns into long-form rows.

        Some Barchart exports provide call metrics to the left of the
        ``Strike`` column and put metrics to the right. Other exports prefix
        column names with ``Call``/``Put``. This helper converts the layout into
        the normalized long format expected by the analyzer.
        """

        normalized_columns = {
            column: re.sub(r"[^a-z0-9]", "", column.lower())
            for column in df.columns
        }

        strike_column = None
        for column, normalized in normalized_columns.items():
            if normalized in {"strike", "strikeprice"}:
                strike_column = column
                break

        if strike_column is None:
            return df

        # Phase 1 unified exports prefix normalized metrics with ``call_`` and
        # ``puts_``. Detect this schema first so downstream tooling receives a
        # consistent long-form view regardless of the upstream format.
        call_prefixed = [
            column for column in df.columns if column.lower().startswith("call_")
        ]
        put_prefixed = [
            column for column in df.columns if column.lower().startswith("puts_")
        ]
        if call_prefixed and put_prefixed:
            call_df = df[[strike_column, *call_prefixed]].copy()
            call_df.rename(
                columns={
                    strike_column: "strike",
                    **{column: column.removeprefix("call_") for column in call_prefixed},
                },
                inplace=True,
            )
            call_df["option_type"] = "call"

            put_df = df[[strike_column, *put_prefixed]].copy()
            put_df.rename(
                columns={
                    strike_column: "strike",
                    **{column: column.removeprefix("puts_") for column in put_prefixed},
                },
                inplace=True,
            )
            put_df["option_type"] = "put"

            combined = pd.concat([call_df, put_df], ignore_index=True)
            for column in combined.columns:
                if column not in {"option_type", "strike"}:
                    combined[column] = pd.to_numeric(combined[column], errors="coerce")
            combined["strike"] = pd.to_numeric(combined["strike"], errors="coerce")
            return combined

        def _to_numeric(series: pd.Series, *, is_percentage: bool = False) -> pd.Series:
            cleaned = (
                series.astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if is_percentage:
                numeric = numeric / 100.0
            return numeric

        prefixed_metrics = {
            "Last": ("last", False),
            "Bid": ("bid", False),
            "Ask": ("ask", False),
            "Change": ("change", False),
            "Volume": ("volume", False),
            "Open Int": ("open_interest", False),
            "Open Interest": ("open_interest", False),
            "IV": ("iv", True),
            "Delta": ("delta", False),
            "Gamma": ("gamma", False),
            "Vega": ("vega", False),
            "Theta": ("theta", False),
            "GEX": ("gex", False),
            "Vanna": ("vanna", False),
        }

        def _build_from_prefix(prefix: str) -> Dict[str, tuple[str, bool]]:
            mapping: Dict[str, tuple[str, bool]] = {}
            normalized_prefix = re.sub(r"[^a-z0-9]", "", prefix.lower())
            for column in df.columns:
                normalized = re.sub(r"[^a-z0-9]", "", column.lower())
                if not normalized.startswith(normalized_prefix):
                    continue
                suffix = normalized[len(normalized_prefix) :]
                for source, meta in prefixed_metrics.items():
                    if suffix == re.sub(r"[^a-z0-9]", "", source.lower()):
                        mapping[column] = meta
                        break
            return mapping

        call_prefixed = _build_from_prefix("Call ")
        put_prefixed = _build_from_prefix("Put ")

        if call_prefixed and put_prefixed:
            call_df = df[[strike_column, *call_prefixed.keys()]].copy()
            call_df.rename(
                columns={
                    strike_column: "strike",
                    **{column: meta[0] for column, meta in call_prefixed.items()},
                },
                inplace=True,
            )
            call_df["option_type"] = "call"

            for column, meta in call_prefixed.items():
                new_name, is_percentage = meta
                if new_name in call_df.columns:
                    call_df[new_name] = _to_numeric(call_df[new_name], is_percentage=is_percentage)

            put_df = df[[strike_column, *put_prefixed.keys()]].copy()
            put_df.rename(
                columns={
                    strike_column: "strike",
                    **{column: meta[0] for column, meta in put_prefixed.items()},
                },
                inplace=True,
            )
            put_df["option_type"] = "put"

            for column, meta in put_prefixed.items():
                new_name, is_percentage = meta
                if new_name in put_df.columns:
                    put_df[new_name] = _to_numeric(put_df[new_name], is_percentage=is_percentage)

            combined = pd.concat([call_df, put_df], ignore_index=True)
            combined["strike"] = pd.to_numeric(combined["strike"], errors="coerce")
            return combined

        # Fallback to classic side-by-side layout with ".1" suffixes.
        base_metrics = {
            "Last": ("last", False),
            "Bid": ("bid", False),
            "Ask": ("ask", False),
            "Change": ("change", False),
            "Volume": ("volume", False),
            "Open Int": ("open_interest", False),
            "Open Interest": ("open_interest", False),
            "IV": ("iv", True),
            "Delta": ("delta", False),
            "Gamma": ("gamma", False),
            "Vega": ("vega", False),
            "Theta": ("theta", False),
        }
        call_columns = {
            column: meta
            for column, meta in base_metrics.items()
            if column in df.columns
        }
        put_columns = {
            f"{column}.1": meta
            for column, meta in base_metrics.items()
            if f"{column}.1" in df.columns
        }

        if not put_columns:
            return df

        call_df = df[[strike_column, *call_columns.keys()]].copy()
        call_df.rename(
            columns={
                strike_column: "strike",
                **{column: meta[0] for column, meta in call_columns.items()},
            },
            inplace=True,
        )
        call_df["option_type"] = "call"

        for column, meta in call_columns.items():
            new_name, is_percentage = meta
            if new_name in call_df.columns:
                call_df[new_name] = _to_numeric(call_df[new_name], is_percentage=is_percentage)

        put_df = df[[strike_column, *put_columns.keys()]].copy()
        put_df.rename(
            columns={
                strike_column: "strike",
                **{column: meta[0] for column, meta in put_columns.items()},
            },
            inplace=True,
        )
        put_df["option_type"] = "put"

        for column, meta in put_columns.items():
            new_name, is_percentage = meta
            if new_name in put_df.columns:
                put_df[new_name] = _to_numeric(put_df[new_name], is_percentage=is_percentage)

        combined = pd.concat([call_df, put_df], ignore_index=True)
        combined["strike"] = pd.to_numeric(combined["strike"], errors="coerce")
        return combined

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Normalize option type column
        type_column = None
        for candidate in ("type", "Type", "option_type", "optionType"):
            if candidate in df.columns:
                type_column = candidate
                break
        if type_column is None:
            raise MissingGreeksError("No option type column found in CSV")
        df[type_column] = df[type_column].astype(str).str.lower().str.strip()
        df[type_column] = df[type_column].replace({"c": "call", "p": "put"})
        df[type_column] = df[type_column].replace({"calls": "call", "puts": "put"})
        df = df[df[type_column].isin(["call", "put"])].copy()
        df.rename(columns={type_column: "option_type"}, inplace=True)

        for column in self.NUMERIC_COLUMNS.intersection(df.columns):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        if "mid" not in df.columns:
            df["mid"] = float("nan")

        if "iv" not in df.columns:
            df["iv"] = float("nan")

        has_open_interest = "open_interest" in df.columns
        if not has_open_interest:
            logger.warning(
                "CSV missing open interest column; defaulting open interest to zero for %s rows",
                len(df),
            )
            df["open_interest"] = 0.0
            df.attrs["open_interest_missing"] = True
        else:
            df.attrs["open_interest_missing"] = False

        missing_mid = df["mid"].isna()
        if missing_mid.any():
            logger.debug("Computing %d synthetic mid prices", missing_mid.sum())
            if "bid" in df.columns and "ask" in df.columns:
                df.loc[missing_mid, "mid"] = (
                    df.loc[missing_mid, ["bid", "ask"]].mean(axis=1)
                )

        required_columns = [
            column
            for column in (
                "strike",
                "gamma",
                "vega",
                "delta",
                "open_interest",
                "gex",
                "vanna",
            )
            if column in df.columns
        ]
        if required_columns:
            before_dropna = len(df)
            df = df.dropna(subset=required_columns)
            dropped_na = before_dropna - len(df)
            if dropped_na:
                logger.debug("Dropped %d rows with non-numeric required fields", dropped_na)

        if has_open_interest:
            before_invalid_oi = len(df)
            df = df[df["open_interest"].notna()]
            dropped_invalid_oi = before_invalid_oi - len(df)
            if dropped_invalid_oi:
                logger.debug("Dropped %d rows with missing open interest values", dropped_invalid_oi)

        return df

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, object]:
        contract_multiplier = self.contract_multiplier
        df = df.copy()
        open_interest_missing = df.attrs.get("open_interest_missing", False)

        call_mask = df["option_type"] == "call"
        put_mask = df["option_type"] == "put"

        has_raw_vanna = "vanna" in df.columns and not df["vanna"].isna().all()
        can_compute_vanna = {
            "delta",
            "vega",
            "open_interest",
        }.issubset(df.columns) and not (
            df["delta"].isna().all() or df["vega"].isna().all()
        )

        if has_raw_vanna:
            df["vanna"] = pd.to_numeric(df["vanna"], errors="coerce").fillna(0.0)
        elif can_compute_vanna:
            df.loc[call_mask, "vanna"] = (
                df.loc[call_mask, "open_interest"]
                * (1 - df.loc[call_mask, "delta"])
                * df.loc[call_mask, "vega"]
                * 100
            )
            df.loc[put_mask, "vanna"] = (
                df.loc[put_mask, "open_interest"]
                * df.loc[put_mask, "delta"]
                * df.loc[put_mask, "vega"]
                * 100
            )
        else:
            raise MissingGreeksError(
                "CSV is missing the data required to compute Vanna (expected delta/vega or a vanna column)."
            )

        gex_factor = contract_multiplier

        has_raw_gex = "gex" in df.columns and not df["gex"].isna().all()
        can_compute_gex = {
            "gamma",
            "open_interest",
        }.issubset(df.columns) and not df["gamma"].isna().all()

        if has_raw_gex:
            df["gex"] = pd.to_numeric(df["gex"], errors="coerce").fillna(0.0)
            if open_interest_missing:
                logger.warning(
                    "Open interest missing from CSV; using provided GEX values without adjustment."
                )
        elif can_compute_gex:
            df["gex"] = df["gamma"] * df["open_interest"] * gex_factor
            if open_interest_missing:
                logger.warning(
                    "Open interest was missing; GEX values are computed assuming zero open interest."
                )
        else:
            if open_interest_missing:
                logger.warning(
                    "Open interest missing from CSV; defaulting GEX to zero for all rows."
                )
                df["gex"] = 0.0
            else:
                raise MissingGreeksError(
                    "CSV is missing the data required to compute GEX (expected gamma or a gex column)."
                )

        totals = df.groupby("option_type")[["gex", "vanna"]].sum()
        call_vanna = float(totals.loc["call", "vanna"]) if "call" in totals.index else 0.0
        put_vanna = float(totals.loc["put", "vanna"]) if "put" in totals.index else 0.0

        calls = df[df["option_type"] == "call"]
        puts = df[df["option_type"] == "put"]

        call_gex = float(calls["gex"].sum()) if not calls.empty else 0.0
        put_gex = float(puts["gex"].sum()) if not puts.empty else 0.0

        gex_by_strike = df.groupby("strike")["gex"].sum().sort_index()
        vanna_by_strike = df.groupby("strike")["vanna"].sum().sort_index()

        can_compute_dex = {"delta", "open_interest"}.issubset(df.columns)
        if can_compute_dex:
            delta_series = pd.to_numeric(df["delta"], errors="coerce").fillna(0.0)
            oi_series = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
            df["dex"] = delta_series * oi_series * contract_multiplier
        else:
            df["dex"] = pd.NA

        can_compute_tex = {"theta", "open_interest"}.issubset(df.columns)
        if can_compute_tex:
            theta_series = pd.to_numeric(df["theta"], errors="coerce").fillna(0.0)
            oi_series = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
            df["tex"] = theta_series * oi_series * contract_multiplier
        else:
            df["tex"] = pd.NA

        df["ivxoi"] = df["iv"].fillna(0) * df["open_interest"].fillna(0)

        agg_kwargs = {
            "gex_sum": ("gex", "sum"),
            "vanna_sum": ("vanna", "sum"),
            "iv_mean": ("iv", "mean"),
            "open_interest_sum": ("open_interest", "sum"),
            "ivxoi_sum": ("ivxoi", "sum"),
        }
        if "volume" in df.columns:
            agg_kwargs["volume_sum"] = ("volume", "sum")
        if can_compute_dex:
            agg_kwargs["dex_sum"] = ("dex", "sum")
        if can_compute_tex:
            agg_kwargs["tex_sum"] = ("tex", "sum")

        grouped = df.groupby(["strike", "option_type"]).agg(**agg_kwargs).reset_index()

        strike_type_metrics: List[StrikeTypeMetric] = []
        for _, row in grouped.iterrows():
            iv_value = float(row["iv_mean"]) if not pd.isna(row["iv_mean"]) else None
            strike_type_metrics.append(
                StrikeTypeMetric(
                    strike=float(row["strike"]),
                    option_type=str(row["option_type"]),
                    next_gex=float(row["gex_sum"]),
                    vanna=float(row["vanna_sum"]),
                    iv=iv_value,
                )
            )

        strike_index = sorted({float(value) for value in df["strike"].unique()})
        summary = pd.DataFrame(index=pd.Index(strike_index, name="strike"))

        call_stats = grouped[grouped["option_type"] == "call"].set_index("strike")
        put_stats = grouped[grouped["option_type"] == "put"].set_index("strike")

        def _side_series(
            stats: pd.DataFrame,
            column: str,
            *,
            default_value: float | object = 0.0,
        ) -> pd.Series:
            if column not in stats.columns:
                if default_value is pd.NA:
                    return pd.Series(pd.NA, index=summary.index, dtype="Float64")
                return pd.Series(default_value, index=summary.index, dtype="Float64")
            series = pd.to_numeric(stats[column], errors="coerce")
            reindexed = series.reindex(summary.index)
            if default_value is not pd.NA:
                reindexed = reindexed.fillna(default_value)
            return reindexed.astype("Float64")

        summary["Call_GEX"] = _side_series(call_stats, "gex_sum")
        summary["Put_GEX"] = _side_series(put_stats, "gex_sum")
        summary["Call_Vanna"] = _side_series(call_stats, "vanna_sum")
        summary["Put_Vanna"] = _side_series(put_stats, "vanna_sum")
        summary["Call_IVxOI"] = _side_series(call_stats, "ivxoi_sum")
        summary["Put_IVxOI"] = _side_series(put_stats, "ivxoi_sum")
        summary["Call_Volume"] = _side_series(call_stats, "volume_sum")
        summary["Put_Volume"] = _side_series(put_stats, "volume_sum")
        summary["Call_OI"] = _side_series(call_stats, "open_interest_sum")
        summary["Put_OI"] = _side_series(put_stats, "open_interest_sum")
        summary["Call_DEX"] = _side_series(call_stats, "dex_sum", default_value=pd.NA)
        summary["Put_DEX"] = _side_series(put_stats, "dex_sum", default_value=pd.NA)
        summary["Call_TEX"] = _side_series(call_stats, "tex_sum", default_value=pd.NA)
        summary["Put_TEX"] = _side_series(put_stats, "tex_sum", default_value=pd.NA)

        summary["Net_GEX"] = summary["Call_GEX"] + summary["Put_GEX"]
        summary["Net_Vanna"] = summary["Call_Vanna"] + summary["Put_Vanna"]
        dex_totals = summary[["Call_DEX", "Put_DEX"]].apply(pd.to_numeric, errors="coerce")
        summary["Net_DEX"] = dex_totals.sum(axis=1, min_count=1).astype("Float64")
        tex_totals = summary[["Call_TEX", "Put_TEX"]].apply(pd.to_numeric, errors="coerce")
        summary["Net_TEX"] = tex_totals.sum(axis=1, min_count=1).astype("Float64")
        for column in ("Call_DEX", "Put_DEX", "Net_DEX", "Call_TEX", "Put_TEX", "Net_TEX"):
            summary[column] = summary[column].round(1)
        summary["IVxOI"] = (summary["Call_IVxOI"] + summary["Put_IVxOI"]).round(1)

        call_ratio_denom = summary["Call_GEX"].replace({0: pd.NA})
        put_ratio_denom = summary["Put_GEX"].replace({0: pd.NA})
        total_ratio_denom = summary["Net_GEX"].replace({0: pd.NA})

        summary["Call_Vanna_Ratio"] = (summary["Call_Vanna"] / call_ratio_denom).astype("Float64")
        summary["Put_Vanna_Ratio"] = (summary["Put_Vanna"] / put_ratio_denom).astype("Float64")
        summary["Vanna_GEX_Total"] = (summary["Net_Vanna"] / total_ratio_denom).astype("Float64")

        median_ivxoi = (
            float(summary["IVxOI"].median(skipna=True))
            if not summary["IVxOI"].dropna().empty
            else None
        )
        summary["Median_IVxOI"] = median_ivxoi

        iv_direction = self.iv_direction
        summary["Energy_Score"] = summary["IVxOI"].apply(_score_energy, median_ivxoi=median_ivxoi)
        summary["Regime"] = summary.apply(
            lambda row: _classify_regime(row["Call_Vanna_Ratio"], row["Put_Vanna_Ratio"], iv_direction),
            axis=1,
        )
        summary["Dealer_Bias"] = summary.apply(
            lambda row: _dealer_bias(row["Call_Vanna_Ratio"], row["Put_Vanna_Ratio"], iv_direction),
            axis=1,
        )

        spot_value = self.spot_price
        if spot_value is None and "underlying_price" in df.columns:
            spot_series = pd.to_numeric(df["underlying_price"], errors="coerce").dropna()
            if not spot_series.empty:
                spot_value = float(spot_series.mean())
        if spot_value is not None and spot_value > 0:
            strike_values = pd.Series(summary.index.astype(float), index=summary.index, dtype="Float64")
            rel_dist = ((strike_values - float(spot_value)).abs() / float(spot_value)).round(4)
            summary["Rel_Dist"] = rel_dist.astype("Float64")
        else:
            summary["Rel_Dist"] = pd.Series(pd.NA, index=summary.index, dtype="Float64")

        timestamp_value = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        summary["DateTime"] = timestamp_value

        summary["Top5_Regime_Energy_Bias"] = ""
        apply_highlight_annotations(summary)
        activity = summary["Call_Volume"] + summary["Put_Volume"]
        if activity.isna().all() or activity.sum() == 0:
            activity = summary["Call_OI"] + summary["Put_OI"]

        top_n = min(5, len(summary))
        if top_n:
            top_indices = activity.fillna(0).astype(float).nlargest(top_n).index
            for strike in top_indices:
                regime = summary.at[strike, "Regime"]
                energy = summary.at[strike, "Energy_Score"]
                bias = summary.at[strike, "Dealer_Bias"]
                if pd.isna(strike):
                    continue
                strike_str = str(int(strike)) if float(strike).is_integer() else f"{float(strike):.2f}".rstrip("0").rstrip(".")
                summary.at[strike, "Top5_Regime_Energy_Bias"] = (
                    f"{strike_str}: {regime} | {energy} | {bias}"
                )

        return {
            "total_gex": float(gex_by_strike.sum()),
            "total_vanna": float(vanna_by_strike.sum()),
            "call_gex": call_gex,
            "put_gex": put_gex,
            "call_vanna": call_vanna,
            "put_vanna": put_vanna,
            "gex_by_strike": gex_by_strike.to_dict(),
            "vanna_by_strike": vanna_by_strike.to_dict(),
            "strike_type_metrics": strike_type_metrics,
            "summary_by_strike": summary,
            "median_ivxoi": median_ivxoi,
            "iv_direction": iv_direction,
        }

    def _write_outputs(self, result: ProcessingResult) -> None:
        output_directory = result.output_directory
        ticker = result.ticker
        expiry = result.expiry
        safe_suffix = f"{ticker}_{expiry}".replace("/", "-")

        json_path = output_directory / f"{safe_suffix}_summary.json"
        csv_path = output_directory / f"{safe_suffix}_per_strike.csv"

        summary_payload = {
            "ticker": ticker,
            "expiry": expiry,
            "total_gex": result.total_gex,
            "total_vanna": result.total_vanna,
            "call_gex": result.call_gex,
            "put_gex": result.put_gex,
            "call_vanna": result.call_vanna,
            "put_vanna": result.put_vanna,
            "gex_by_strike": result.gex_by_strike,
            "vanna_by_strike": result.vanna_by_strike,
            "source_path": str(result.source_path),
            "call_vanna_ratio_by_strike": result.call_vanna_ratio_by_strike,
            "put_vanna_ratio_by_strike": result.put_vanna_ratio_by_strike,
            "vanna_gex_total_by_strike": result.vanna_gex_total_by_strike,
            "energy_score_by_strike": result.energy_score_by_strike,
            "regime_by_strike": result.regime_by_strike,
            "dealer_bias_by_strike": result.dealer_bias_by_strike,
            "ivxoi_by_strike": result.ivxoi_by_strike,
            "rel_dist_by_strike": result.rel_dist_by_strike,
            "top5_bias_summary": result.top5_bias_summary,
            "median_ivxoi": result.median_ivxoi,
            "iv_direction": result.iv_direction,
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        logger.info("Wrote summary JSON to %s", json_path)

        summary_df = result.strike_summary_df.copy().reset_index().rename(columns={"index": "strike"})
        per_strike_df = pd.DataFrame(
            {
                "strike": summary_df["strike"].astype(float),
                "gex": summary_df["Net_GEX"].astype(float),
                "vanna": summary_df["Net_Vanna"].astype(float),
                "call_vanna_ratio": summary_df["Call_Vanna_Ratio"],
                "put_vanna_ratio": summary_df["Put_Vanna_Ratio"],
                "vanna_gex_total": summary_df["Vanna_GEX_Total"],
                "ivxoi": summary_df["IVxOI"],
                "median_ivxoi": summary_df["Median_IVxOI"],
                "energy_score": summary_df["Energy_Score"],
                "regime": summary_df["Regime"],
                "dealer_bias": summary_df["Dealer_Bias"],
                "iv_direction": result.iv_direction,
                "rel_dist": summary_df["Rel_Dist"],
                "top5_regime_energy_bias": summary_df["Top5_Regime_Energy_Bias"],
            }
        )
        per_strike_df.to_csv(csv_path, index=False)
        logger.info("Wrote per-strike CSV to %s", csv_path)

        self._update_highlight_log(result, safe_suffix)

        self._persist_to_database(result)

        if self.create_charts and plt is not None:
            self._create_charts(result, output_directory / f"{safe_suffix}_charts.png")
        elif self.create_charts:
            logger.warning(
                "matplotlib is not available; skipping chart generation for %s",
                result.source_path,
            )

    def _update_highlight_log(self, result: ProcessingResult, safe_suffix: str) -> None:
        summary = result.strike_summary_df
        if summary.empty:
            logger.debug("No strike data available for highlight logging")
            return


        highlight_columns = [column for column in HIGHLIGHT_METRIC_MAP.values() if column in summary.columns]
        if not highlight_columns:
            logger.debug("No highlight columns present in strike summary; skipping highlight log")
            return

        safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", result.ticker)
        highlight_log_dir = result.output_directory / "highlight_logs"
        highlight_log_path = highlight_log_dir / f"{safe_ticker}_highlight_log.csv"

        existing_df: pd.DataFrame | None = None
        if highlight_log_path.exists():
            existing_df = pd.read_csv(
                highlight_log_path,
                thousands=",",
                na_values=[""],
                keep_default_na=True,
            )

        highlight_mask = summary[highlight_columns].fillna("").ne("").any(axis=1)
        if not highlight_mask.any():
            logger.debug(
                "No highlights detected for %s %s; skipping highlight log",
                result.ticker,
                result.expiry,
            )
            return

        summary_with_strike = summary.copy()
        strike_series = pd.to_numeric(pd.Series(summary_with_strike.index, index=summary_with_strike.index), errors="coerce")
        summary_with_strike["_strike"] = strike_series
        summary_with_strike = summary_with_strike.loc[summary_with_strike["_strike"].notna()]
        if summary_with_strike.empty:
            logger.debug("Highlight rows contained no numeric strikes; skipping highlight log")
            return

        summary_with_strike["_strike_rounded"] = summary_with_strike["_strike"].round(2)

        highlighted_indices = highlight_mask[highlight_mask].index
        highlighted_rows = summary_with_strike.loc[
            summary_with_strike.index.intersection(highlighted_indices)
        ].copy()
        if highlighted_rows.empty:
            logger.debug("Highlight rows contained no numeric strikes; skipping highlight log")
            return

        timestamp_series = highlighted_rows.get("DateTime")
        if timestamp_series is None:
            timestamp_value = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            timestamp_series = pd.Series([timestamp_value] * len(highlighted_rows), index=highlighted_rows.index)
        else:
            timestamp_series = timestamp_series.astype(str)

        log_length = len(highlighted_rows)
        log_data: Dict[str, List[object]] = {
            "Ticker": [result.ticker] * log_length,
            "Expiry": [result.expiry] * log_length,
            "Run_Timestamp": timestamp_series.tolist(),
            "Strike": highlighted_rows["_strike_rounded"].astype(float).tolist(),
        }

        for metric, highlight_column in HIGHLIGHT_METRIC_MAP.items():
            if metric not in highlighted_rows.columns or highlight_column not in highlighted_rows.columns:
                log_data[metric] = [pd.NA] * log_length
                continue
            values = pd.to_numeric(highlighted_rows[metric], errors="coerce")
            log_data[metric] = values.tolist()

        log_df = pd.DataFrame(log_data)
        log_df = log_df.sort_values("Strike", ascending=False).reset_index(drop=True)

        highlight_log_dir.mkdir(parents=True, exist_ok=True)
        if existing_df is not None:
            combined_df = pd.concat([existing_df, log_df], ignore_index=True, sort=False)
            combined_df = combined_df.reindex(columns=log_df.columns)
        else:
            combined_df = log_df

        combined_df["Strike"] = pd.to_numeric(combined_df["Strike"], errors="coerce")
        combined_df = combined_df.sort_values(
            by=["Ticker", "Strike", "Run_Timestamp"],
            ascending=[True, True, False],
            ignore_index=True,
        )

        numeric_columns = [
            column
            for column in combined_df.columns
            if column not in {"Ticker", "Expiry", "Run_Timestamp"}
        ]

        def _format_numeric(value: float) -> str:
            if pd.isna(value):
                return ""
            formatted = f"{value:,.2f}"
            if "." in formatted:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted

        formatted_df = combined_df.copy()
        for column in numeric_columns:
            numeric_source = combined_df[column]
            numeric_series = pd.to_numeric(
                numeric_source.astype(str).str.replace(",", ""),
                errors="coerce",
            ).round(2)
            formatted_df[column] = numeric_series.apply(_format_numeric)

        formatted_df.to_csv(
            highlight_log_path,
            index=False,
            na_rep="",
        )
        logger.info("Updated highlight log CSV at %s", highlight_log_path)

    def _persist_to_database(self, result: ProcessingResult) -> None:
        if not result.strike_type_metrics:
            logger.debug("No strike-level metrics available for database persistence")
            return

        database_path = result.output_directory / "option_metrics.sqlite"
        try:
            database = AnalyticsDatabase(database_path)
            database.record_metrics(result.ticker, result.strike_type_metrics)
            logger.info("Persisted strike metrics to %s", database_path)
        except Exception:  # pragma: no cover - surfaced via CLI logging
            logger.exception("Failed to persist metrics to database at %s", database_path)

    def _create_charts(self, result: ProcessingResult, chart_path: Path) -> None:
        if plt is None:  # pragma: no cover - guarded in caller
            raise RuntimeError("matplotlib is required for chart generation")
        ordered_keys = sorted(result.gex_by_strike.keys(), key=float)
        strikes = [float(s) for s in ordered_keys]
        gex_values = [float(result.gex_by_strike[s]) for s in ordered_keys]
        vanna_values = [float(result.vanna_by_strike.get(s, 0.0)) for s in ordered_keys]

        plt.style.use("ggplot")
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(strikes, gex_values, marker="o", linestyle="-")
        axes[0].set_ylabel("GEX")
        axes[0].set_title(f"Gamma Exposure vs Strike ({result.ticker} {result.expiry})")

        axes[1].plot(strikes, vanna_values, marker="o", linestyle="-", color="tab:orange")
        axes[1].set_ylabel("Vanna")
        axes[1].set_xlabel("Strike")
        axes[1].set_title("Vanna vs Strike")

        fig.tight_layout()
        fig.savefig(chart_path, dpi=200)
        plt.close(fig)
        logger.info("Wrote chart to %s", chart_path)
