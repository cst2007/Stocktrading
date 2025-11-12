"""Tests for filesystem discovery in :mod:`barchart.analyzer`."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barchart.analyzer import BarchartOptionsAnalyzer


def _build_sample_csv(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "option_type": ["call", "put"],
            "strike": [100.0, 100.0],
            "gamma": [0.1, 0.2],
            "vega": [0.15, 0.25],
            "delta": [0.55, 0.45],
            "open_interest": [150, 200],
            "gex": [12.5, 18.2],
            "vanna": [8.1, 9.4],
            "underlying_price": [100.0, 100.0],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_process_path_detects_uppercase_csv(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    csv_path = input_dir / "sample-options-exp-2024-01-19.CSV"
    _build_sample_csv(csv_path)

    output_dir = tmp_path / "output"
    analyzer = BarchartOptionsAnalyzer(create_charts=False, spot_price=100.0)

    results = analyzer.process_path(input_dir, output_dir)

    assert len(results) == 1
    assert results[0].source_path == csv_path.resolve()
