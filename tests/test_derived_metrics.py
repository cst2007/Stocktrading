from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barchart.derived_metrics import compute_derived_metrics


def _build_sample_frame() -> pd.DataFrame:
    strikes = [100, 101, 102, 103, 104, 105]
    call_vanna = [5.0, 25.0, 7.0, 25.0, 12.0, 18.0]
    put_vanna = [4.0, 8.0, 25.0, 7.0, 10.0, 30.0]
    call_gex = [10.0, 10.0, 10.0, 10.0, 0.0, 10.0]
    put_gex = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    call_open_interest = [100, 110, 120, 130, 140, 150]
    put_open_interest = [120, 130, 140, 150, 160, 170]

    call_iv = [0.10, 0.50, 0.20, 0.10, 0.10, 0.20]
    put_iv = [0.10, 0.20, 0.80, 0.10, 0.10, 0.15]

    call_volume = [600, 500, 400, 300, 200, 100]
    put_volume = [580, 450, 380, 260, 180, 90]

    frame = pd.DataFrame(
        {
            "Strike": strikes,
            "call_vanna": call_vanna,
            "puts_vanna": put_vanna,
            "net_vanna": [c + p for c, p in zip(call_vanna, put_vanna)],
            "call_gex": call_gex,
            "puts_gex": put_gex,
            "net_gex": [c + p for c, p in zip(call_gex, put_gex)],
            "call_delta": [0.45, 0.55, 0.40, 0.50, 0.35, 0.60],
            "puts_delta": [0.55, 0.45, 0.60, 0.40, 0.65, 0.35],
            "call_open_interest": call_open_interest,
            "puts_open_interest": put_open_interest,
            "call_iv": call_iv,
            "puts_iv": put_iv,
            "call_oi_iv": [oi * iv for oi, iv in zip(call_open_interest, call_iv)],
            "puts_oi_iv": [oi * iv for oi, iv in zip(put_open_interest, put_iv)],
            "call_volume": call_volume,
            "puts_volume": put_volume,
            "Spot": [102.0] * len(strikes),
        }
    )

    return frame


def test_ratios_and_regime_classification():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
    )

    gamma_pin = metrics.loc[metrics["Strike"] == 100].iloc[0]
    assert pytest.approx(gamma_pin["Call_Vanna_Ratio"], rel=1e-6) == 0.5
    assert gamma_pin["Regime"] == "Gamma Pin"

    zero_call = metrics.loc[metrics["Strike"] == 104].iloc[0]
    assert pd.isna(zero_call["Call_Vanna_Ratio"])

    pre_earnings = metrics.loc[metrics["Strike"] == 101].iloc[0]
    assert pre_earnings["Call_Vanna_Ratio"] > 2
    assert pre_earnings["Regime"] == "Pre-Earnings Fade"

    vol_drift_up = metrics.loc[metrics["Strike"] == 102].iloc[0]
    assert vol_drift_up["Regime"] == "Vol Drift Up"

    rel_dist = metrics.loc[metrics["Strike"] == 101, "Rel_Dist"].iloc[0]
    expected_rel_dist = round(abs(101 - 102) / 102, 4)
    assert rel_dist == expected_rel_dist


def test_directional_bias_switches_with_iv_direction():
    df = _build_sample_frame()
    metrics_down = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="down",
    )

    post_earnings = metrics_down.loc[metrics_down["Strike"] == 102].iloc[0]
    assert post_earnings["Regime"] == "Post-Earnings Vanna Rally"
    assert post_earnings["Dealer_Bias"] == "Dealer Buying → Bullish Drift"

    bearish_fade = metrics_down.loc[metrics_down["Strike"] == 101].iloc[0]
    assert bearish_fade["Dealer_Bias"] == "Dealer Selling → Bearish Fade"

    vol_drift_down = metrics_down.loc[metrics_down["Strike"] == 103].iloc[0]
    assert vol_drift_down["Regime"] == "Vol Drift Down"


def test_unknown_direction_is_preserved_and_neutral():
    df = _build_sample_frame()
    metrics_unknown = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="unknown",
    )

    sample_row = metrics_unknown.loc[metrics_unknown["Strike"] == 102].iloc[0]
    assert sample_row["IV_Direction"] == "unknown"
    assert sample_row["Regime"] == "Vol Drift Up"
    assert sample_row["Dealer_Bias"] == "Neutral / Mean Reversion"


def test_top5_column_only_populated_for_top_five():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
    )

    populated = metrics["Top5_Regime_Energy_Bias"].replace("", pd.NA).dropna()
    assert len(populated) == 5
    lowest_activity_row = metrics.loc[metrics["Strike"] == 105].iloc[0]
    assert lowest_activity_row["Top5_Regime_Energy_Bias"] == ""


def test_totals_row_is_appended_when_requested():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
        include_totals_row=True,
    )

    totals_row = metrics.iloc[-1]
    assert totals_row["Strike"] == "Total"
    assert totals_row["Call_Vanna_Highlight"] == ""
    assert totals_row["Put_Vanna_Highlight"] == ""

    data_only = metrics.iloc[:-1]
    expected_net_gex_total = pd.to_numeric(data_only["Net_GEX"], errors="coerce").sum()
    assert totals_row["Net_GEX"] == pytest.approx(round(expected_net_gex_total, 2))


def test_columns_can_be_dropped_from_output():
    df = _build_sample_frame()
    exclusions = {
        "Energy_Score",
        "Regime",
        "Dealer_Bias",
        "IV_Direction",
        "Rel_Dist",
        "Top5_Regime_Energy_Bias",
        "Call_IV",
        "Put_IV",
        "Median_IVxOI",
    }
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
        drop_columns=exclusions,
        include_totals_row=True,
    )

    for column in exclusions:
        assert column not in metrics.columns
    assert "Net_GEX" in metrics.columns


def test_put_vanna_highlight_marks_top_strikes():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
    )

    highlighted = metrics.loc[metrics["Put_Vanna_Highlight"] == "highlight"]
    top_put_values = metrics["Put_Vanna"].nlargest(3)
    assert set(highlighted["Strike"]) == set(metrics.loc[top_put_values.index, "Strike"])


def test_invalid_direction_raises_error():
    df = _build_sample_frame()
    with pytest.raises(ValueError):
        compute_derived_metrics(
            df,
            calculation_time=datetime.now(timezone.utc),
            spot_price=102.0,
            iv_direction="sideways",
        )
