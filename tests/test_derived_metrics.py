from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barchart.derived_metrics import _format_strike, compute_derived_metrics


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

    call_theta = [-0.05, -0.08, -0.03, -0.04, -0.02, -0.01]
    put_theta = [-0.02, -0.03, -0.06, -0.01, -0.04, -0.05]

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
            "call_theta": call_theta,
            "puts_theta": put_theta,
            "call_open_interest": call_open_interest,
            "puts_open_interest": put_open_interest,
            "call_iv": call_iv,
            "puts_iv": put_iv,
            "call_oi_iv": [oi * iv for oi, iv in zip(call_open_interest, call_iv)],
            "puts_oi_iv": [oi * iv for oi, iv in zip(put_open_interest, put_iv)],
            "call_volume": call_volume,
            "puts_volume": put_volume,
            "call_gamma": [gex / (oi * 100) for gex, oi in zip(call_gex, call_open_interest)],
            "puts_gamma": [gex / (oi * 100) for gex, oi in zip(put_gex, put_open_interest)],
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
    expected_call_tex = -0.05 * 100 * 100
    assert gamma_pin["Call_TEX"] == pytest.approx(round(expected_call_tex, 1))

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


def test_dgex_dspot_uses_deep_itm_strikes_and_ranks_extremes():
    data = pd.DataFrame(
        {
            "Strike": [100, 105, 110],
            "call_vanna": [0.0, 0.0, 0.0],
            "puts_vanna": [0.0, 0.0, 0.0],
            "net_vanna": [0.0, 0.0, 0.0],
            "call_gex": [1.0, 1.0, 1.0],
            "puts_gex": [0.25, 0.25, 0.25],
            "net_gex": [0.75, 0.75, 0.75],
            "call_delta": [0.9, 0.6, 0.4],
            "puts_delta": [0.1, 0.2, 0.85],
            "call_theta": [0.0, 0.0, 0.0],
            "puts_theta": [0.0, 0.0, 0.0],
            "call_open_interest": [10, 10, 10],
            "puts_open_interest": [5, 5, 5],
            "call_iv": [0.1, 0.1, 0.1],
            "puts_iv": [0.1, 0.1, 0.1],
            "call_oi_iv": [1.0, 1.0, 1.0],
            "puts_oi_iv": [0.5, 0.5, 0.5],
            "call_volume": [0, 0, 0],
            "puts_volume": [0, 0, 0],
            "call_gamma": [0.001, 0.001, 0.001],
            "puts_gamma": [0.0005, 0.0005, 0.0005],
            "Spot": [105.0, 105.0, 105.0],
        }
    )

    metrics = compute_derived_metrics(
        data,
        calculation_time=datetime.now(timezone.utc),
        spot_price=105.0,
        iv_direction="up",
    )

    dgex_values = metrics.set_index("Strike")["dGEX/dSpot"].dropna()

    assert dgex_values.loc[100.0] == pytest.approx(450.0)
    assert dgex_values.loc[105.0] == pytest.approx(472.5)
    assert dgex_values.loc[110.0] == pytest.approx(495.0)

    assert (
        metrics.loc[metrics["Strike"] == 110, "dGEX/dSpot Rank"].iloc[0]
        == "Top 1: 110"
    )
    assert (
        metrics.loc[metrics["Strike"] == 105, "dGEX/dSpot Rank"].iloc[0]
        == "Top 2: 105"
    )
    assert (
        metrics.loc[metrics["Strike"] == 100, "dGEX/dSpot Rank"].iloc[0]
        == "Top 3: 100"
    )


def test_dgex_dspot_evaluates_strikes_within_fifteen_steps_of_spot():
    strike_values = list(range(80, 120))
    data = pd.DataFrame(
        {
            "Strike": strike_values,
            "call_vanna": [0.0] * len(strike_values),
            "puts_vanna": [0.0] * len(strike_values),
            "net_vanna": [0.0] * len(strike_values),
            "call_gex": [1.0] * len(strike_values),
            "puts_gex": [0.25] * len(strike_values),
            "net_gex": [0.75] * len(strike_values),
            "call_delta": [0.5] * len(strike_values),
            "puts_delta": [0.5] * len(strike_values),
            "call_theta": [0.0] * len(strike_values),
            "puts_theta": [0.0] * len(strike_values),
            "call_open_interest": [10] * len(strike_values),
            "puts_open_interest": [5] * len(strike_values),
            "call_iv": [0.1] * len(strike_values),
            "puts_iv": [0.1] * len(strike_values),
            "call_oi_iv": [1.0] * len(strike_values),
            "puts_oi_iv": [0.5] * len(strike_values),
            "call_volume": [0] * len(strike_values),
            "puts_volume": [0] * len(strike_values),
            "call_gamma": [0.001] * len(strike_values),
            "puts_gamma": [0.0005] * len(strike_values),
            "Spot": [100.0] * len(strike_values),
        }
    )

    metrics = compute_derived_metrics(
        data,
        calculation_time=datetime.now(timezone.utc),
        spot_price=100.0,
        iv_direction="down",
    )

    dgex_values = metrics.set_index("Strike")["dGEX/dSpot"]

    populated = dgex_values.dropna()
    assert len(populated) == 31
    assert populated.loc[100.0] == pytest.approx(6000.0)
    assert pd.isna(dgex_values.loc[84.0])
    assert pd.isna(dgex_values.loc[116.0])


def test_ivxoi_columns_positioned_and_ranked():
    strikes = list(range(100, 108))
    call_open_interest = [80 + 10 * idx for idx in range(len(strikes))]
    put_open_interest = [100 + 10 * idx for idx in range(len(strikes))]
    call_iv = [0.1] * len(strikes)
    put_iv = [0.2] * len(strikes)

    data = pd.DataFrame(
        {
            "Strike": strikes,
            "call_vanna": [5.0] * len(strikes),
            "puts_vanna": [3.0] * len(strikes),
            "net_vanna": [8.0] * len(strikes),
            "call_gex": [1.0] * len(strikes),
            "puts_gex": [0.5] * len(strikes),
            "net_gex": [1.5] * len(strikes),
            "call_delta": [0.5] * len(strikes),
            "puts_delta": [0.5] * len(strikes),
            "call_theta": [-0.01] * len(strikes),
            "puts_theta": [-0.02] * len(strikes),
            "call_open_interest": call_open_interest,
            "puts_open_interest": put_open_interest,
            "call_iv": call_iv,
            "puts_iv": put_iv,
            "call_oi_iv": [oi * iv for oi, iv in zip(call_open_interest, call_iv)],
            "puts_oi_iv": [oi * iv for oi, iv in zip(put_open_interest, put_iv)],
            "call_volume": [0] * len(strikes),
            "puts_volume": [0] * len(strikes),
            "call_gamma": [gex / (oi * 100) for gex, oi in zip([1.0] * len(strikes), call_open_interest)],
            "puts_gamma": [gex / (oi * 100) for gex, oi in zip([0.5] * len(strikes), put_open_interest)],
            "Spot": [104.0] * len(strikes),
        }
    )

    metrics = compute_derived_metrics(
        data,
        calculation_time=datetime.now(timezone.utc),
        spot_price=104.0,
        iv_direction="up",
    )

    columns = list(metrics.columns)
    dgex_rank_idx = columns.index("dGEX/dSpot Rank")
    assert columns[dgex_rank_idx + 1 : dgex_rank_idx + 7] == [
        "Call_IVxOI",
        "Put_IVxOI",
        "IVxOI",
        "IVxOI Rank",
        "Call_IVxOI_Rank",
        "Put_IVxOI_Rank",
    ]

    ivxoi_ranks = metrics.set_index("Strike")["IVxOI Rank"]
    assert ivxoi_ranks.loc[100] == ""
    assert ivxoi_ranks.loc[107] == "Rank 1: 107"
    assert ivxoi_ranks.loc[106] == "Rank 2: 106"
    assert ivxoi_ranks.loc[101] == "Rank 7: 101"


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
    assert totals_row["Net_GEX_Highlight"] == ""
    assert totals_row["DEX_highlight"] == ""
    assert totals_row["TEX_highlight"] == ""
    assert totals_row["Call_IVxOI_Rank"] == ""
    assert totals_row["Put_IVxOI_Rank"] == ""

    data_only = metrics.iloc[:-1]
    expected_net_gex_total = pd.to_numeric(data_only["Net_GEX"], errors="coerce").sum()
    assert totals_row["Net_GEX"] == pytest.approx(round(expected_net_gex_total, 2))
    expected_net_tex_total = pd.to_numeric(data_only["Net_TEX"], errors="coerce").sum()
    assert totals_row["Net_TEX"] == pytest.approx(round(expected_net_tex_total, 2))


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


def test_put_vex_columns_populated_in_spx_mode():
    df = _build_sample_frame()
    extra_row = {
        "Strike": 106,
        "call_vanna": 18.0,
        "puts_vanna": -30.0,
        "net_vanna": -12.0,
        "call_gex": 10.0,
        "puts_gex": 10.0,
        "net_gex": 20.0,
        "call_delta": 0.6,
        "puts_delta": 0.4,
        "call_theta": -0.02,
        "puts_theta": -0.05,
        "call_open_interest": 160,
        "puts_open_interest": 180,
        "call_iv": 0.12,
        "puts_iv": 0.12,
        "call_oi_iv": 19.2,
        "puts_oi_iv": 21.6,
        "call_volume": 80,
        "puts_volume": 85,
        "Spot": 102.0,
        "call_gamma": 10.0 / (160 * 100),
        "puts_gamma": 10.0 / (180 * 100),
    }
    df = pd.concat([df, pd.DataFrame([extra_row])], ignore_index=True)

    df["puts_vanna"] = [-5.0, -15.0, -8.0, -12.0, -25.0, -10.0, -30.0]
    df["call_vanna"] = [10.0, 15.0, 8.0, 25.0, 12.0, 20.0, 18.0]
    df["net_vanna"] = df["call_vanna"] + df["puts_vanna"]
    df["call_iv"] = [0.10, 0.50, 0.20, 0.10, 0.10, 0.20, 0.12]
    df["puts_iv"] = [0.10, 0.20, 0.80, 0.10, 0.10, 0.15, 0.12]
    df["call_open_interest"] = [100, 110, 120, 130, 140, 150, 160]
    df["puts_open_interest"] = [120, 130, 140, 150, 160, 170, 180]
    df["call_oi_iv"] = df["call_open_interest"] * df["call_iv"]
    df["puts_oi_iv"] = df["puts_open_interest"] * df["puts_iv"]

    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
        include_put_vex=True,
    )

    assert metrics.columns[1:11].tolist() == [
        "Put VEX",
        "Put VEX Rank",
        "Call VEX",
        "Call VEX Rank",
        "Net_DEX",
        "DEX_highlight",
        "Net_GEX",
        "Net_GEX_Highlight",
        "dGEX/dSpot",
        "dGEX/dSpot Rank",
    ]

    put_vex_value = metrics.loc[metrics["Strike"] == 100, "Put VEX"].iloc[0]
    expected_put_vex = df.loc[0, "puts_vanna"] * df.loc[0, "puts_iv"] * df.loc[0, "puts_open_interest"]
    assert put_vex_value == pytest.approx(round(expected_put_vex, 2))

    call_vex_value = metrics.loc[metrics["Strike"] == 100, "Call VEX"].iloc[0]
    expected_call_vex = df.loc[0, "call_vanna"] * df.loc[0, "call_iv"] * df.loc[0, "call_open_interest"]
    assert call_vex_value == pytest.approx(round(expected_call_vex, 2))

    ranked_rows = metrics.loc[metrics["Put VEX Rank"] != ""]
    assert len(ranked_rows) == 7
    assert metrics.loc[metrics["Strike"] == 102, "Put VEX Rank"].iloc[0] == "Rank 1: 102"
    assert metrics.loc[metrics["Strike"] == 106, "Put VEX Rank"].iloc[0] == "Rank 2: 106"
    assert metrics.loc[metrics["Strike"] == 104, "Put VEX Rank"].iloc[0] == "Rank 3: 104"
    assert metrics.loc[metrics["Strike"] == 101, "Put VEX Rank"].iloc[0] == "Rank 4: 101"
    assert metrics.loc[metrics["Strike"] == 105, "Put VEX Rank"].iloc[0] == "Rank 5: 105"
    assert metrics.loc[metrics["Strike"] == 103, "Put VEX Rank"].iloc[0] == "Rank 6: 103"
    assert metrics.loc[metrics["Strike"] == 100, "Put VEX Rank"].iloc[0] == "Rank 7: 100"

    ranked_call_rows = metrics.loc[metrics["Call VEX Rank"] != ""]
    assert len(ranked_call_rows) == 5
    assert metrics.loc[metrics["Strike"] == 101, "Call VEX Rank"].iloc[0] == "Rank 1: 101"
    assert metrics.loc[metrics["Strike"] == 105, "Call VEX Rank"].iloc[0] == "Rank 2: 105"
    assert metrics.loc[metrics["Strike"] == 103, "Call VEX Rank"].iloc[0] == "Rank 3: 103"
    assert metrics.loc[metrics["Strike"] == 106, "Call VEX Rank"].iloc[0] == "Rank 4: 106"
    assert metrics.loc[metrics["Strike"] == 102, "Call VEX Rank"].iloc[0] == "Rank 5: 102"


def test_put_vanna_highlight_marks_top_strikes():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
    )

    highlighted = metrics.loc[metrics["Put_Vanna_Highlight"] != ""]
    top_put_indices = metrics["Put_Vanna"].nsmallest(4).index

    assert set(highlighted.index) == set(top_put_indices)

    for rank, idx in enumerate(top_put_indices, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert (
            metrics.at[idx, "Put_Vanna_Highlight"]
            == f"Top {rank} : {strike_value}"
        )


def test_dex_highlight_marks_top_and_bottom_strikes():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
    )

    highlighted = metrics.loc[metrics["DEX_highlight"] != ""]
    top_dex_indices = metrics["Net_DEX"].nlargest(4).index
    bottom_dex_indices = metrics["Net_DEX"].nsmallest(4).index
    bottom_dex_exclusive = [idx for idx in bottom_dex_indices if idx not in set(top_dex_indices)]

    expected_indices = set(top_dex_indices).union(set(bottom_dex_exclusive))
    assert set(highlighted.index) == expected_indices

    for rank, idx in enumerate(top_dex_indices, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert metrics.at[idx, "DEX_highlight"] == f"Top {rank} : {strike_value}"

    for rank, idx in enumerate(bottom_dex_exclusive, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert metrics.at[idx, "DEX_highlight"] == f"Bottom {rank} : {strike_value}"


def test_net_gex_highlight_marks_top_and_bottom_strikes():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
    )

    highlighted = metrics.loc[metrics["Net_GEX_Highlight"] != ""]
    top_gex_indices = metrics["Net_GEX"].nlargest(4).index
    bottom_gex_indices = metrics["Net_GEX"].nsmallest(4).index
    bottom_gex_exclusive = [idx for idx in bottom_gex_indices if idx not in set(top_gex_indices)]

    expected_indices = set(top_gex_indices).union(set(bottom_gex_exclusive))
    assert set(highlighted.index) == expected_indices

    for rank, idx in enumerate(top_gex_indices, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert metrics.at[idx, "Net_GEX_Highlight"] == f"Top {rank} : {strike_value}"

    for rank, idx in enumerate(bottom_gex_exclusive, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert metrics.at[idx, "Net_GEX_Highlight"] == f"Bottom {rank} : {strike_value}"


def test_tex_highlight_marks_top_strikes():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
    )

    highlighted = metrics.loc[metrics["TEX_highlight"] != ""]
    top_tex_indices = metrics["Net_TEX"].nsmallest(5).index

    assert set(highlighted.index) == set(top_tex_indices)

    for rank, idx in enumerate(top_tex_indices, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert metrics.at[idx, "TEX_highlight"] == f"Top {rank} : {strike_value}"


def test_call_and_put_tex_highlights_mark_most_negative():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
    )

    call_highlighted = metrics.loc[metrics["Call_TEX_Highlight"] != ""]
    top_call_indices = metrics["Call_TEX"].nsmallest(5).index
    assert set(call_highlighted.index) == set(top_call_indices)
    for rank, idx in enumerate(top_call_indices, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert metrics.at[idx, "Call_TEX_Highlight"] == f"Top {rank} : {strike_value}"

    put_highlighted = metrics.loc[metrics["Put_TEX_Highlight"] != ""]
    top_put_indices = metrics["Put_TEX"].nsmallest(5).index
    assert set(put_highlighted.index) == set(top_put_indices)
    for rank, idx in enumerate(top_put_indices, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert metrics.at[idx, "Put_TEX_Highlight"] == f"Top {rank} : {strike_value}"


def test_ivxoi_highlights_mark_top_values():
    df = _build_sample_frame()
    metrics = compute_derived_metrics(
        df,
        calculation_time=datetime.now(timezone.utc),
        spot_price=102.0,
        iv_direction="up",
    )

    call_highlighted = metrics.loc[metrics["Call_IVxOI_Rank"] != ""]
    put_highlighted = metrics.loc[metrics["Put_IVxOI_Rank"] != ""]

    top_call_indices = metrics["Call_IVxOI"].nlargest(4).index
    top_put_indices = metrics["Put_IVxOI"].nlargest(4).index

    assert set(call_highlighted.index) == set(top_call_indices)
    assert set(put_highlighted.index) == set(top_put_indices)

    for rank, idx in enumerate(top_call_indices, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert metrics.at[idx, "Call_IVxOI_Rank"] == f"Top {rank} : {strike_value}"

    for rank, idx in enumerate(top_put_indices, start=1):
        strike_value = _format_strike(metrics.at[idx, "Strike"])
        assert metrics.at[idx, "Put_IVxOI_Rank"] == f"Top {rank} : {strike_value}"


def test_invalid_direction_raises_error():
    df = _build_sample_frame()
    with pytest.raises(ValueError):
        compute_derived_metrics(
            df,
            calculation_time=datetime.now(timezone.utc),
            spot_price=102.0,
            iv_direction="sideways",
        )
