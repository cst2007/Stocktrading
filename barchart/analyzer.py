"""Core analytics engine for the Barchart Options Analyzer."""

from __future__ import annotations

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

logger = logging.getLogger(__name__)


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


@dataclass
class StrikeTypeMetric:
    strike: float
    option_type: str
    next_gex: float
    vanna: float
    iv: float | None


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
    }

    def __init__(
        self,
        contract_multiplier: float = 100.0,
        create_charts: bool = True,
        spot_price: float | None = None,
    ) -> None:
        self.contract_multiplier = contract_multiplier
        self.create_charts = create_charts
        self.spot_price = spot_price

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_path(self, path: Path, output_directory: Path) -> List[ProcessingResult]:
        """Process a single CSV or every CSV within a directory."""

        path = path.expanduser().resolve()
        output_directory = output_directory.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        if path.is_file():
            return [self._process_file(path, output_directory)]

        if not path.exists():
            raise FileNotFoundError(f"Input path {path} does not exist")

        results: List[ProcessingResult] = []
        for csv_path in sorted(path.glob("**/*.csv")):
            try:
                results.append(self._process_file(csv_path, output_directory))
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
        )

        self._write_outputs(result)
        return result

    def _parse_metadata_from_filename(self, filename: str) -> (str, str):
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

        if "Strike" not in df.columns:
            return df

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
            "IV": ("iv", True),
            "Delta": ("delta", False),
            "Gamma": ("gamma", False),
            "Vega": ("vega", False),
            "Theta": ("theta", False),
        }

        def _build_from_prefix(prefix: str) -> Dict[str, tuple[str, bool]]:
            mapping: Dict[str, tuple[str, bool]] = {}
            for source, meta in prefixed_metrics.items():
                column_name = f"{prefix}{source}"
                if column_name in df.columns:
                    mapping[column_name] = meta
            return mapping

        call_prefixed = _build_from_prefix("Call ")
        put_prefixed = _build_from_prefix("Put ")

        if call_prefixed and put_prefixed:
            call_df = df[["Strike", *call_prefixed.keys()]].copy()
            call_df.rename(
                columns={
                    "Strike": "strike",
                    **{column: meta[0] for column, meta in call_prefixed.items()},
                },
                inplace=True,
            )
            call_df["option_type"] = "call"

            for column, meta in call_prefixed.items():
                new_name, is_percentage = meta
                if new_name in call_df.columns:
                    call_df[new_name] = _to_numeric(call_df[new_name], is_percentage=is_percentage)

            put_df = df[["Strike", *put_prefixed.keys()]].copy()
            put_df.rename(
                columns={
                    "Strike": "strike",
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
            "IV": ("iv", True),
            "Delta": ("delta", False),
            "Gamma": ("gamma", False),
            "Vega": ("vega", False),
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

        call_df = df[["Strike", *call_columns.keys()]].copy()
        call_df.rename(
            columns={
                "Strike": "strike",
                **{column: meta[0] for column, meta in call_columns.items()},
            },
            inplace=True,
        )
        call_df["option_type"] = "call"

        for column, meta in call_columns.items():
            new_name, is_percentage = meta
            if new_name in call_df.columns:
                call_df[new_name] = _to_numeric(call_df[new_name], is_percentage=is_percentage)

        put_df = df[["Strike", *put_columns.keys()]].copy()
        put_df.rename(
            columns={
                "Strike": "strike",
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
        for candidate in ("type", "option_type", "optionType"):
            if candidate in df.columns:
                type_column = candidate
                break
        if type_column is None:
            raise KeyError("No option type column found in CSV")
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

        missing_required = [column for column in ("delta", "gamma", "vega") if column not in df.columns]
        if missing_required:
            raise KeyError(
                "Missing required columns in CSV: " + ", ".join(sorted(missing_required))
            )

        has_open_interest = "open_interest" in df.columns
        if not has_open_interest:
            logger.warning(
                "CSV missing open interest column; defaulting open interest to zero for %s rows",
                len(df),
            )
            df["open_interest"] = 0.0

        missing_mid = df["mid"].isna()
        if missing_mid.any():
            logger.debug("Computing %d synthetic mid prices", missing_mid.sum())
            if "bid" in df.columns and "ask" in df.columns:
                df.loc[missing_mid, "mid"] = (
                    df.loc[missing_mid, ["bid", "ask"]].mean(axis=1)
                )

        required_columns = [
            column
            for column in ("strike", "gamma", "vega", "delta", "open_interest")
            if column in df.columns
        ]
        if required_columns:
            df = df.dropna(subset=required_columns)

        if has_open_interest:
            df = df[df["open_interest"] > 0]

        return df

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, object]:
        if self.spot_price is None:
            raise ValueError("spot_price must be provided before computing metrics")

        contract_multiplier = self.contract_multiplier
        spot_price = float(self.spot_price)

        df = df.copy()

        call_mask = df["option_type"] == "call"
        put_mask = df["option_type"] == "put"

        df.loc[call_mask, "vanna"] = (
            (1 - df.loc[call_mask, "delta"]) * df.loc[call_mask, "vega"] * 100
        )
        df.loc[put_mask, "vanna"] = (
            df.loc[put_mask, "delta"] * df.loc[put_mask, "vega"] * 100
        )

        gex_factor = (contract_multiplier * (spot_price**2)) / 10000.0

        df["gex"] = df["gamma"] * df["open_interest"] * gex_factor
        df.loc[df["option_type"] == "put", "gex"] *= -1

        totals = df.groupby("option_type")[["gex", "vanna"]].sum()
        call_vanna = float(totals.loc["call", "vanna"]) if "call" in totals.index else 0.0
        put_vanna = float(totals.loc["put", "vanna"]) if "put" in totals.index else 0.0

        calls = df[df["option_type"] == "call"]
        puts = df[df["option_type"] == "put"]

        call_gex = float((calls["gamma"] * calls["open_interest"] * gex_factor).sum()) if not calls.empty else 0.0
        put_gex = float((puts["gamma"] * puts["open_interest"] * gex_factor).sum()) if not puts.empty else 0.0

        gex_by_strike = df.groupby("strike")["gex"].sum().sort_index()
        vanna_by_strike = df.groupby("strike")["vanna"].sum().sort_index()

        grouped = (
            df.groupby(["strike", "option_type"])
            .agg(
                next_gex=("gex", "sum"),
                vanna=("vanna", "sum"),
                iv=("iv", "mean"),
            )
            .reset_index()
        )

        strike_type_metrics: List[StrikeTypeMetric] = []
        for _, row in grouped.iterrows():
            iv_value = float(row["iv"]) if not pd.isna(row["iv"]) else None
            strike_type_metrics.append(
                StrikeTypeMetric(
                    strike=float(row["strike"]),
                    option_type=str(row["option_type"]),
                    next_gex=float(row["next_gex"]),
                    vanna=float(row["vanna"]),
                    iv=iv_value,
                )
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
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        logger.info("Wrote summary JSON to %s", json_path)

        strikes = sorted(result.gex_by_strike.keys(), key=float)
        per_strike_df = pd.DataFrame(
            {
                "strike": strikes,
                "gex": [result.gex_by_strike[s] for s in strikes],
                "vanna": [result.vanna_by_strike.get(s, 0.0) for s in strikes],
            }
        )
        per_strike_df.to_csv(csv_path, index=False)
        logger.info("Wrote per-strike CSV to %s", csv_path)

        self._persist_to_database(result)

        if self.create_charts and plt is not None:
            self._create_charts(result, output_directory / f"{safe_suffix}_charts.png")
        elif self.create_charts:
            logger.warning(
                "matplotlib is not available; skipping chart generation for %s",
                result.source_path,
            )

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
