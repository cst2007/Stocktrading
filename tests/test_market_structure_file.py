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
