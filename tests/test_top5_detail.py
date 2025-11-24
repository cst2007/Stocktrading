from __future__ import annotations

import pandas as pd

from barchart.analyzer import _compute_top5_detail


def test_compute_top5_detail_prioritization():
    index = pd.Index([100, 105, 110, 115, 120, 90], dtype=float)
    summary = pd.DataFrame(
        {
            "Call_Volume": [10, 0, 20, 25, 15, 12],
            "Put_Volume": [5, 0, 30, 15, 15, 10],
            "Call_OI": [50, 120, 80, 75, 60, 55],
            "Put_OI": [40, 80, 70, 65, 55, 45],
            "Energy_Score": ["Moderate", "High", "High", "High", "Moderate", "High"],
            "Regime": [
                "Gamma Pin",
                "Gamma Pin",
                "Vol Drift Up",
                "Vol Drift Down",
                "Transition Zone",
                "Transition Zone",
            ],
            "Dealer_Bias": [
                "Neutral / Mean Reversion",
                "Dealer Selling → Bearish Fade",
                "Dealer Buying → Bullish Drift",
                "Dealer Selling → Bearish Fade",
                "Neutral / Mean Reversion",
                "Neutral / Mean Reversion",
            ],
            "Distance_To_Spot": [-3, 2, 7, 12, 17, -13],
        },
        index=index,
    )

    detail = _compute_top5_detail(summary, 103.0)

    assert detail["Primary_Fade_Level"] == 105
    assert detail["Primary_Long_Drift_Level"] == 110
    assert detail["Primary_Short_Drift_Level"] == 115
    assert detail["Flip_Zone"] == 90
    assert detail["Nearest_Magnet"] == 90
    assert detail["Secondary_Magnet"] == 105
    assert len(detail["Top5_Detail"]) == 5
    classifications = {row["Strike"]: row["Classification"] for row in detail["Top5_Detail"]}
    assert "Short_Drift_Zone" in classifications[115]
