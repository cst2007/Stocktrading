"""Helpers for discovering and processing pairs of Barchart CSV exports."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from .analyzer import BarchartOptionsAnalyzer, ProcessingResult
from .ai_insights import AIInsightsConfigurationError, generate_ai_insight
from .combiner import COMBINED_CSV_HEADER, combine_option_files
from .derived_metrics import DERIVED_CSV_HEADER, compute_derived_metrics

logger = logging.getLogger(__name__)

_PAIR_PATTERN = re.compile(
    r"^(?P<ticker>[a-zA-Z0-9$]+)-(?P<kind>options|volatility-greeks)-exp-"
    r"(?P<expiry>\d{4}-\d{2}-\d{2})-weekly-(?P<week>\d+)-strikes(?P<suffix>.*)\.csv$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class OptionFilePair:
    """Representation of a side-by-side/greeks export pair."""

    key: str
    ticker: str
    expiry: str
    side_by_side_path: Path
    greeks_path: Path
    upload_time: datetime

    @property
    def label(self) -> str:
        if self.expiry != "UNKNOWN":
            return f"{self.ticker} {self.expiry}"
        return self.ticker

    @property
    def batch_key(self) -> str:
        return self.key

    def to_dict(self, *, relative_to: Path | None = None) -> Dict[str, object]:
        def _format_path(path: Path) -> str:
            if relative_to is not None:
                try:
                    return str(path.relative_to(relative_to))
                except ValueError:
                    pass
            return str(path)

        return {
            "id": self.key,
            "batch_key": self.batch_key,
            "ticker": self.ticker,
            "expiry": self.expiry,
            "label": self.label,
            "side_by_side": _format_path(self.side_by_side_path),
            "greeks": _format_path(self.greeks_path),
            "upload_time": self.upload_time.isoformat().replace("+00:00", "Z"),
        }


def discover_pairs(input_directory: Path) -> List[OptionFilePair]:
    """Return every pair of side-by-side and greeks CSV files in ``input_directory``."""

    input_directory = input_directory.expanduser().resolve()

    grouped: Dict[str, Dict[str, object]] = {}

    for csv_path in sorted(input_directory.glob("*.csv")):
        name_lower = csv_path.name.lower()
        if "combined" in name_lower:
            continue
        if "side-by-side" not in name_lower and "volatility-greeks" not in name_lower:
            continue

        match = _PAIR_PATTERN.match(csv_path.name)
        if not match:
            logger.debug("Ignoring unmatched file name: %s", csv_path.name)
            continue

        ticker = match.group("ticker").lstrip("$").upper()
        expiry = match.group("expiry")
        week = match.group("week")
        suffix = match.group("suffix") or ""
        normalized_suffix = re.sub(r"-side-by-side", "", suffix, flags=re.IGNORECASE)
        kind = match.group("kind").lower()

        group_key = f"{ticker}__{expiry}__{week}{normalized_suffix.lower()}"
        entry = grouped.setdefault(
            group_key,
            {
                "ticker": ticker,
                "expiry": expiry,
                "week": week,
                "suffix": normalized_suffix,
                "paths": {},
            },
        )
        entry["paths"][kind] = csv_path

    pairs: List[OptionFilePair] = []
    for key, info in grouped.items():
        paths: Dict[str, Path] = info["paths"]
        options_path = paths.get("options")
        greeks_path = paths.get("volatility-greeks")

        if options_path is None or greeks_path is None:
            missing_kind = "volatility" if options_path else "options"
            logger.warning(
                "[%s] %s %s skipped: missing %s file",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                info["ticker"],
                info["expiry"],
                missing_kind,
            )
            continue

        upload_timestamp = datetime.fromtimestamp(
            max(options_path.stat().st_mtime, greeks_path.stat().st_mtime),
            tz=timezone.utc,
        )
        batch_key = (
            f"{info['ticker']}_{info['expiry']}"
            f"_{upload_timestamp.strftime('%Y%m%dT%H%M%SZ')}"
        )

        pairs.append(
            OptionFilePair(
                key=batch_key,
                ticker=info["ticker"],
                expiry=info["expiry"],
                side_by_side_path=options_path,
                greeks_path=greeks_path,
                upload_time=upload_timestamp,
            )
        )

    return sorted(pairs, key=lambda pair: pair.upload_time, reverse=True)


def process_pair(
    pair: OptionFilePair,
    *,
    spot_price: float,
    output_directory: Path,
    processed_directory: Path,
    contract_multiplier: float = 100.0,
    create_charts: bool = False,
    enable_insights: bool = False,
) -> Dict[str, object]:
    """Combine and analyze ``pair``, then move the inputs into ``processed_directory``."""

    if spot_price is None:
        raise ValueError("spot_price must be provided when processing a file pair")

    output_directory = output_directory.expanduser().resolve()
    processed_directory = processed_directory.expanduser().resolve()

    output_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)

    combined_df = combine_option_files(
        pair.side_by_side_path,
        pair.greeks_path,
        spot_price=spot_price,
        contract_multiplier=contract_multiplier,
    )

    combined_filename = _derive_combined_filename(pair)
    combined_path = output_directory / combined_filename
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(combined_path, index=False, header=COMBINED_CSV_HEADER)

    derived_dir = output_directory / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    calculation_time = datetime.now(timezone.utc)
    derived_df = compute_derived_metrics(combined_df, calculation_time=calculation_time)
    safe_ticker = pair.ticker.replace("/", "-") or "unknown"
    safe_expiry = (
        pair.expiry.replace("/", "-") if pair.expiry != "UNKNOWN" else "unknown"
    )
    derived_filename = (
        f"derived_metrics_{safe_ticker}_{safe_expiry}_"
        f"{calculation_time.strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    derived_path = derived_dir / derived_filename
    derived_df.to_csv(derived_path, index=False, header=DERIVED_CSV_HEADER)

    insights_dir = derived_dir / "insights"
    insights_info: Dict[str, object] | None = None
    if enable_insights:
        try:
            insights_info = generate_ai_insight(
                combined_df=combined_df,
                derived_df=derived_df,
                derived_path=derived_path,
                ticker=pair.ticker,
                expiry=pair.expiry,
                insights_dir=insights_dir,
            )
        except AIInsightsConfigurationError as exc:
            logger.warning("Skipping AI insight generation: %s", exc)
        except Exception:  # pragma: no cover - surfaced via CLI logging
            logger.exception("Failed to generate AI insight for %s", derived_path)

    analyzer_output_dir = output_directory / "analysis"
    analyzer_output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = BarchartOptionsAnalyzer(
        contract_multiplier=contract_multiplier,
        create_charts=create_charts,
        spot_price=float(spot_price),
    )

    results = analyzer.process_path(combined_path, analyzer_output_dir)
    summaries = _summaries_from_results(results)

    moved_files = [
        _move_to_processed(pair.side_by_side_path, processed_directory),
        _move_to_processed(pair.greeks_path, processed_directory),
    ]

    return {
        "pair": pair.to_dict(relative_to=pair.side_by_side_path.parent),
        "combined_csv": str(combined_path),
        "moved_files": [str(path) for path in moved_files],
        "derived_csv": str(derived_path),
        "summaries": summaries,
        "insights": insights_info,
    }


def _summaries_from_results(results: Sequence[ProcessingResult]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for result in results:
        safe_suffix = f"{result.ticker}_{result.expiry}".replace("/", "-")
        summary_path = result.output_directory / f"{safe_suffix}_summary.json"
        per_strike_path = result.output_directory / f"{safe_suffix}_per_strike.csv"
        chart_path = result.output_directory / f"{safe_suffix}_charts.png"

        summary_info: Dict[str, object] = {
            "ticker": result.ticker,
            "expiry": result.expiry,
            "summary_json": str(summary_path),
            "per_strike_csv": str(per_strike_path),
        }

        if chart_path.exists():
            summary_info["chart"] = str(chart_path)

        summaries.append(summary_info)

    return summaries


def _move_to_processed(source: Path, destination_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Source file {source} does not exist")

    destination = destination_dir / source.name
    if destination.exists():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        destination = destination_dir / f"{source.stem}_{timestamp}{source.suffix}"

    shutil.move(str(source), destination)
    return destination


def _derive_combined_filename(pair: OptionFilePair) -> str:
    expiry = pair.expiry.replace("/", "-") if pair.expiry != "UNKNOWN" else "unknown"
    ticker = pair.ticker.replace("/", "-") or "unknown"
    return f"unified_{ticker}_{expiry}.csv"


__all__ = ["OptionFilePair", "discover_pairs", "process_pair"]
