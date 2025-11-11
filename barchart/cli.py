"""Command line interface for the Barchart Options Analyzer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

from .analyzer import BarchartOptionsAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute aggregate Vanna and GEX metrics from Barchart options CSV files.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a Barchart CSV file or a directory containing CSV files.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="./output",
        help="Directory where summary files and charts will be written.",
    )
    parser.add_argument(
        "--contract-multiplier",
        type=float,
        default=100.0,
        help="Contract size used in exposure calculations (default: 100).",
    )
    parser.add_argument(
        "--spot-price",
        type=float,
        required=True,
        help="Underlying spot price to use in exposure calculations.",
    )
    parser.add_argument(
        "--iv-direction",
        choices=["up", "down", "unknown"],
        required=True,
        help="Observed implied volatility trend for the current session.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Disable generation of PNG charts for strike curves.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser


def _format_number(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.2f}"


def run_cli(args: Iterable[str] | None = None) -> List[str]:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    logging.basicConfig(
        level=getattr(logging, parsed.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    analyzer = BarchartOptionsAnalyzer(
        contract_multiplier=parsed.contract_multiplier,
        create_charts=not parsed.no_charts,
        spot_price=parsed.spot_price,
        iv_direction=parsed.iv_direction,
    )

    input_path = Path(parsed.input)
    output_directory = Path(parsed.out)

    results = analyzer.process_path(input_path, output_directory)
    if not results:
        parser.exit(1, "No CSV files processed.\n")

    summaries: List[str] = []
    for result in results:
        summary = (
            f"Processed {result.source_path.name}: "
            f"Total GEX={_format_number(result.total_gex)}, "
            f"Total Vanna={_format_number(result.total_vanna)}"
        )
        logging.info(summary)
        summaries.append(summary)

    return summaries


def main() -> None:  # pragma: no cover - CLI entry point
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
