from pathlib import Path

from barchart.pair_processor import _write_market_structure_file


def test_write_market_structure_file(tmp_path: Path) -> None:
    derived_path = tmp_path / "derived_metrics_sample.csv"
    derived_path.write_text("dummy", encoding="utf-8")

    playbook = {
        "next_step": "Look for Gamma Box bottom → prepare for long entry.",
        "useful_metrics": ["GEX curvature (dGEX/dSpot)", "VEX: positive VEX supports melt-up"],
        "avoid": "Shorts. Overthinking. This is clean.",
    }

    path = _write_market_structure_file(
        derived_path,
        market_state="Regime Flip",
        market_state_description="Gamma flipped with bearish pressure",
        market_state_components={"GEX_location": "OTM", "DEX_zero": 123.45},
        market_state_playbook=playbook,
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
