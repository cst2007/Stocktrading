from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barchart.analyzer import BarchartOptionsAnalyzer


def test_debug_mode_limits_rows_and_logs_steps(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "demo-options-exp-2024-01-19.csv"
    output_dir = tmp_path / "output"

    frame = pd.DataFrame(
        {
            "option_type": ["call", "put"],
            "strike": [100.0, 100.0],
            "gamma": [0.1, 0.2],
            "delta": [0.5, -0.4],
            "vega": [0.25, 0.35],
            "open_interest": [10, 20],
            "gex": [5.0, 15.0],
            "vanna": [1.0, 2.0],
            "underlying_price": [100.0, 100.0],
        }
    )
    frame.to_csv(csv_path, index=False)

    analyzer = BarchartOptionsAnalyzer(create_charts=False, debug_mode=True, spot_price=100.0)

    with caplog.at_level("INFO"):
        results = analyzer.process_path(csv_path, output_dir)

    assert len(results) == 1
    result = results[0]

    # Only the first row should contribute to totals when debug mode is on.
    assert result.total_gex == pytest.approx(5.0)
    assert result.total_vanna == pytest.approx(1.0)

    messages = "\n".join(caplog.messages)
    assert "limiting analysis to the first row" in messages
    assert "Final per-row exposures" in messages

