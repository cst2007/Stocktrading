from pathlib import Path

import pandas as pd

from barchart.pair_processor import _write_market_structure_file


def test_write_market_structure_file(tmp_path: Path) -> None:
    derived_path = tmp_path / "derived_metrics_sample.csv"
    derived_path.write_text("dummy", encoding="utf-8")

    derived_df = pd.DataFrame(
        {
            "Strike": [4300, 4400, 4200],
            "Net_GEX": [10_000_000, -25_000_000, 30_000_000],
        }
    )

    playbook = {
        "next_step": "Look for Gamma Box bottom → prepare for long entry.",
        "useful_metrics": ["GEX curvature (dGEX/dSpot)", "VEX: positive VEX supports melt-up"],
        "avoid": "Shorts. Overthinking. This is clean.",
    }

    top5_detail = {
        "Primary_Fade_Level": 4400,
        "Primary_Long_Drift_Level": None,
        "Primary_Short_Drift_Level": 4200,
        "Flip_Zone": None,
        "Nearest_Magnet": 4300,
        "Secondary_Magnet": None,
        "Top5_Detail": [
            {
                "Strike": 4400,
                "Classification": "Fade_Zone, Magnet",
                "Energy_Score": "High",
                "Regime": "Gamma Pin",
                "Dealer_Bias": "Bearish Fade",
                "Distance_To_Spot": 50,
            },
            {
                "Strike": 4200,
                "Classification": "Short_Drift_Zone",
                "Energy_Score": "High",
                "Regime": "Vol Drift Down",
                "Dealer_Bias": "Bearish Fade",
                "Distance_To_Spot": -150,
            },
        ],
    }

    path = _write_market_structure_file(
        derived_path,
        market_state="Regime Flip",
        market_state_description="Gamma flipped with bearish pressure",
        market_state_components={"GEX_location": "OTM", "DEX_zero": 123.45},
        market_state_playbook=playbook,
        derived_df=derived_df,
        ticker="SPX",
        spot_price=4350,
        vex_direction=1,
        tex_direction=-1,
        gamma_box_high=4350,
        gamma_box_low=4200,
        breakout_up=True,
        breakout_down=False,
        vex_dir_box_high=1,
        vex_dir_box_low=-1,
        tex_dir_box_high=1,
        tex_dir_box_low=-1,
        top5_detail=top5_detail,
    )

    assert path is not None
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Market State: Regime Flip" in content
    assert "Description: Gamma flipped with bearish pressure" in content
    assert "- DEX_zero: 123.45" in content
    assert "- GEX_location: OTM" in content
    assert "Playbook:" in content
    assert "Next Step: Look for Gamma Box bottom → prepare for long entry." in content
    assert "Useful Metrics:" in content
    assert "- GEX curvature (dGEX/dSpot)" in content
    assert "Avoid: Shorts. Overthinking. This is clean." in content
    assert "VEX Direction:" in content
    assert "- VEX_dir: 1" in content
    assert "Upside fuel" in content
    assert "TEX Direction:" in content
    assert "- TEX_dir: -1" in content
    assert "Downward pressure" in content
    assert "Execution:" in content
    assert "- Gamma_Box_High: 4350" in content
    assert "- Gamma_Box_Low: 4200" in content
    assert "Breakout_Up: True" in content
    assert "Breakout_Down: False" in content
    assert "VEX_dir_Box_high: 1" in content
    assert "Downside fuel" in content
    assert "TEX_dir_Box_high: 1" in content
    assert "Slow downside drift" in content
    assert "Magnets:" in content
    assert "Primary: 4200" in content
    assert "Levels (max 10): 4200" in content
    assert "Direction: 1 (Market pulled UP)" in content
    assert "Threshold: 29,300,000" in content
    assert "Top 5 Detail:" in content
    assert "Primary_Fade_Level: 4400" in content
    assert "Primary_Short_Drift_Level: 4200" in content
    assert "Nearest_Magnet: 4300" in content
    assert "Fade_Zone, Magnet | Regime: Gamma Pin | Energy: High | Dealer Bias: Bearish Fade" in content
    assert "4200: Short_Drift_Zone | Regime: Vol Drift Down" in content


def test_write_market_structure_file_skips_when_absent(tmp_path: Path) -> None:
    derived_path = tmp_path / "derived_metrics_sample.csv"
    derived_path.write_text("dummy", encoding="utf-8")

    path = _write_market_structure_file(
        derived_path,
        market_state="",
        market_state_description="",
        market_state_components={},
    )

    assert path is None
    assert not any(tmp_path.glob("*_market_structure.txt"))
