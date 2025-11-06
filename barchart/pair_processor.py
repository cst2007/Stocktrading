"""Helpers for discovering and processing pairs of Barchart CSV exports."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from .analyzer import BarchartOptionsAnalyzer, ProcessingResult
from .combiner import COMBINED_CSV_HEADER, combine_option_files

_PAIR_PATTERN = re.compile(
    r"^(?P<prefix>.+?)-(?:options|volatility-greeks)-exp-(?P<suffix>.+)\.csv$",
    re.IGNORECASE,
)
_EXPIRY_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(slots=True)
class OptionFilePair:
    """Representation of a side-by-side/greeks export pair."""

    key: str
    ticker: str
    expiry: str
    side_by_side_path: Path
    greeks_path: Path
    last_updated: datetime

    @property
    def label(self) -> str:
        if self.expiry != "UNKNOWN":
            return f"{self.ticker} {self.expiry}"
        return self.ticker

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
            "ticker": self.ticker,
            "expiry": self.expiry,
            "label": self.label,
            "side_by_side": _format_path(self.side_by_side_path),
            "greeks": _format_path(self.greeks_path),
            "last_modified": self.last_updated.isoformat(),
        }


def discover_pairs(input_directory: Path) -> List[OptionFilePair]:
    """Return every pair of side-by-side and greeks CSV files in ``input_directory``."""

    input_directory = input_directory.expanduser().resolve()

    side_files: Dict[str, Path] = {}
    greeks_files: Dict[str, Path] = {}
    metadata: Dict[str, Dict[str, str]] = {}

    for csv_path in sorted(input_directory.glob("*.csv")):
        name_lower = csv_path.name.lower()
        if "combined" in name_lower:
            continue
        if "side-by-side" not in name_lower and "volatility-greeks" not in name_lower:
            continue

        match = _PAIR_PATTERN.match(csv_path.name)
        if not match:
            continue

        prefix = match.group("prefix")
        suffix = match.group("suffix")
        normalized_suffix = suffix.replace("side-by-side-", "")
        key = f"{prefix.lower()}__{normalized_suffix.lower()}"

        ticker = prefix.lstrip("$").upper()
        expiry_match = _EXPIRY_PATTERN.search(suffix)
        expiry = expiry_match.group(1) if expiry_match else "UNKNOWN"

        metadata[key] = {"ticker": ticker, "expiry": expiry}

        if "side-by-side" in name_lower:
            side_files[key] = csv_path
        elif "volatility-greeks" in name_lower:
            greeks_files[key] = csv_path

    pairs: List[OptionFilePair] = []
    for key in sorted(set(side_files).intersection(greeks_files)):
        side_path = side_files[key]
        greeks_path = greeks_files[key]
        info = metadata[key]
        last_modified = datetime.fromtimestamp(
            max(side_path.stat().st_mtime, greeks_path.stat().st_mtime)
        )

        pairs.append(
            OptionFilePair(
                key=key,
                ticker=info["ticker"],
                expiry=info["expiry"],
                side_by_side_path=side_path,
                greeks_path=greeks_path,
                last_updated=last_modified,
            )
        )

    return pairs


def process_pair(
    pair: OptionFilePair,
    *,
    spot_price: float,
    output_directory: Path,
    processed_directory: Path,
    contract_multiplier: float = 100.0,
    create_charts: bool = False,
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

    combined_filename = _derive_combined_filename(pair.side_by_side_path.name)
    combined_path = output_directory / combined_filename
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(combined_path, index=False, header=COMBINED_CSV_HEADER)

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
        "summaries": summaries,
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


def _derive_combined_filename(side_by_side_name: str) -> str:
    if "side-by-side" in side_by_side_name:
        return side_by_side_name.replace("side-by-side", "combined")
    stem, dot, suffix = side_by_side_name.partition(".")
    if not dot:
        return f"{side_by_side_name}_combined"
    return f"{stem}_combined.{suffix}"


__all__ = ["OptionFilePair", "discover_pairs", "process_pair"]
