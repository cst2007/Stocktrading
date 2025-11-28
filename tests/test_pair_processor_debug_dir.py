from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barchart.pair_processor import OptionFilePair, process_pair


def test_v2_process_pair_uses_debug_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    processed_dir = tmp_path / "processed"

    options_csv = tmp_path / "spx-options-exp-2025-11-03-weekly-1-strikes.csv"
    greeks_csv = tmp_path / "spx-volatility-greeks-exp-2025-11-03-weekly-1-strikes.csv"
    options_csv.write_text("placeholder")
    greeks_csv.write_text("placeholder")

    pair = OptionFilePair(
        key="pair-1",
        ticker="SPX",
        expiry="2025-11-03",
        side_by_side_path=options_csv,
        greeks_path=greeks_csv,
        upload_time=datetime.now(timezone.utc),
        version="v2",
    )

    captured: dict[str, Path | None] = {}

    def fake_run_exposure_pipeline(
        side_by_side_path: Path,
        greeks_path: Path,
        config,
        *,
        output_dir: Path,
        debug_dir: Path | None = None,
    ):
        from barchart.v2_pipeline import ExposureOutputs

        captured["debug_dir"] = debug_dir
        return ExposureOutputs(
            core_path=output_dir / "core.csv",
            side_path=output_dir / "side.csv",
            reactivity_path=output_dir / "reactivity.csv",
            derived_path=output_dir / "derived.csv",
            premium_path=output_dir / "premium.csv",
        )

    monkeypatch.setattr("barchart.v2_pipeline.run_exposure_pipeline", fake_run_exposure_pipeline)

    process_pair(
        pair,
        spot_price=100.0,
        iv_direction="up",
        output_directory=output_dir,
        processed_directory=processed_dir,
        debug_mode=True,
    )

    assert captured["debug_dir"] == output_dir.resolve() / "debug"
    assert (processed_dir / options_csv.name).exists()
    assert (processed_dir / greeks_csv.name).exists()
