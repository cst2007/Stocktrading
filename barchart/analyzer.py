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
        r"\$(?P<ticker>[a-zA-Z0-9]+)-options-exp-(?P<expiry>\d{4}-\d{2}-\d{2}).*\.csv$"
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
        logger.debug("Loaded %d rows", len(df))
        return df

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

        missing_mid = df["mid"].isna()
        if missing_mid.any():
            logger.debug("Computing %d synthetic mid prices", missing_mid.sum())
            if "bid" in df.columns and "ask" in df.columns:
                df.loc[missing_mid, "mid"] = (
                    df.loc[missing_mid, ["bid", "ask"]].mean(axis=1)
                )

        df = df.dropna(subset=["strike", "gamma", "vega", "delta", "open_interest"])
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

        return {
            "total_gex": float(gex_by_strike.sum()),
            "total_vanna": float(vanna_by_strike.sum()),
            "call_gex": call_gex,
            "put_gex": put_gex,
            "call_vanna": call_vanna,
            "put_vanna": put_vanna,
            "gex_by_strike": gex_by_strike.to_dict(),
            "vanna_by_strike": vanna_by_strike.to_dict(),
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

        if self.create_charts and plt is not None:
            self._create_charts(result, output_directory / f"{safe_suffix}_charts.png")
        elif self.create_charts:
            logger.warning(
                "matplotlib is not available; skipping chart generation for %s",
                result.source_path,
            )

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
