import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barchart.analyzer import BarchartOptionsAnalyzer


def test_open_interest_defaults_to_one_when_missing(caplog):
    analyzer = BarchartOptionsAnalyzer(create_charts=False)
    df = pd.DataFrame(
        {
            "option_type": ["call", "put"],
            "strike": [100, 100],
            "gamma": [0.01, 0.02],
            "delta": [0.6, -0.4],
            "vega": [0.2, 0.25],
            "theta": [-0.01, -0.02],
        }
    )

    processed = analyzer._preprocess(df)

    assert processed.attrs["open_interest_missing"] is True
    assert processed["open_interest"].tolist() == [1.0, 1.0]

    with caplog.at_level("WARNING"):
        metrics = analyzer._compute_metrics(processed)

    expected_gex = (0.01 + 0.02) * analyzer.contract_multiplier
    assert metrics["total_gex"] == pytest.approx(expected_gex)

    expected_call_vanna = (1 - 0.6) * 0.2 * analyzer.contract_multiplier
    expected_put_vanna = (-0.4) * 0.25 * analyzer.contract_multiplier
    assert metrics["total_vanna"] == pytest.approx(
        expected_call_vanna + expected_put_vanna
    )

    assert any(
        "one contract per row for GEX calculations" in message
        for message in caplog.messages
    )
