"""Helpers for discovering and processing pairs of Barchart CSV exports."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from numbers import Number
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from .analyzer import BarchartOptionsAnalyzer, ProcessingResult
from .ai_insights import AIInsightsConfigurationError, generate_ai_insight
from .combiner import COMBINED_CSV_HEADER, combine_option_files
from .derived_metrics import compute_derived_metrics
from derived.OptionSelling_premium_components import build_premium_components

logger = logging.getLogger(__name__)

# Accept both weekly and monthly exports. The trailing numeric segment is treated
# generically so existing grouping logic continues to work for either cadence.
_PAIR_PATTERN = re.compile(
    r"^(?P<ticker>[a-zA-Z0-9$]+)-(?P<kind>options|volatility-greeks)-exp-"
    r"(?P<expiry>\d{4}-\d{2}-\d{2})-(?:weekly|monthly)-(?P<week>\d+)-strikes(?P<suffix>.*)\.csv$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class OptionFilePair:
    """Representation of an options/greeks export pair."""

    key: str
    ticker: str
    expiry: str
    side_by_side_path: Path
    greeks_path: Path
    upload_time: datetime
    version: str = "v1"

    @property
    def label(self) -> str:
        if self.expiry != "UNKNOWN":
            return f"{self.ticker} {self.expiry}"
        return self.ticker

    @property
    def batch_key(self) -> str:
        return self.key

    def to_dict(self, *, relative_to: Path | None = None) -> Dict[str, object]:
        def _format_path(path: Path) -> str:
            if relative_to is not None:
                try:
                    return str(path.relative_to(relative_to))
                except ValueError:
                    pass
            return str(path)

        return {
            "id": self.key,
            "batch_key": self.batch_key,
            "ticker": self.ticker,
            "expiry": self.expiry,
            "label": self.label,
            "side_by_side": _format_path(self.side_by_side_path),
            "greeks": _format_path(self.greeks_path),
            "upload_time": self.upload_time.isoformat().replace("+00:00", "Z"),
            "version": self.version,
        }


def discover_pairs(input_directory: Path) -> List[OptionFilePair]:
    """Return every pair of side-by-side and greeks CSV files in ``input_directory``."""

    input_directory = input_directory.expanduser().resolve()

    grouped: Dict[str, Dict[str, object]] = {}

    simple_pattern = re.compile(
        r"^(?P<ticker>[a-zA-Z0-9$]+)-(?P<kind>options|volatility-greeks)-exp-(?P<expiry>\d{4}-\d{2}-\d{2})(?P<suffix>.*)\.csv$",
        re.IGNORECASE,
    )

    for csv_path in sorted(input_directory.glob("*.csv")):
        name_lower = csv_path.name.lower()
        if "combined" in name_lower:
            continue

        match = _PAIR_PATTERN.match(csv_path.name)
        version = "v1"
        if not match:
            match = simple_pattern.match(csv_path.name)
            version = "v2" if match else "v1"
        if not match:
            logger.debug("Ignoring unmatched file name: %s", csv_path.name)
            continue

        ticker = match.group("ticker").lstrip("$").upper()
        expiry = match.group("expiry")
        suffix = match.groupdict().get("suffix") or ""
        week = match.groupdict().get("week", "")
        normalized_suffix = re.sub(r"-side-by-side", "", suffix, flags=re.IGNORECASE)
        kind = match.group("kind").lower()

        group_key = f"{ticker}__{expiry}__{week}{normalized_suffix.lower()}"
        entry = grouped.setdefault(
            group_key,
            {
                "ticker": ticker,
                "expiry": expiry,
                "week": week,
                "suffix": normalized_suffix,
                "paths": {},
                "version": version,
            },
        )
        entry["paths"][kind] = csv_path
        entry["version"] = version or entry.get("version", "v1")

    pairs: List[OptionFilePair] = []
    for key, info in grouped.items():
        paths: Dict[str, Path] = info["paths"]
        options_path = paths.get("options")
        greeks_path = paths.get("volatility-greeks")

        if options_path is None or greeks_path is None:
            missing_kind = "volatility" if options_path else "options"
            logger.warning(
                "[%s] %s %s skipped: missing %s file",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                info["ticker"],
                info["expiry"],
                missing_kind,
            )
            continue

        upload_timestamp = datetime.fromtimestamp(
            max(options_path.stat().st_mtime, greeks_path.stat().st_mtime),
            tz=timezone.utc,
        )
        batch_key = (
            f"{info['ticker']}_{info['expiry']}"
            f"_{upload_timestamp.strftime('%Y%m%dT%H%M%SZ')}"
        )

        pairs.append(
            OptionFilePair(
                key=batch_key,
                ticker=info["ticker"],
                expiry=info["expiry"],
                side_by_side_path=options_path,
                greeks_path=greeks_path,
                upload_time=upload_timestamp,
                version=info.get("version", "v1"),
            )
        )

    return sorted(pairs, key=lambda pair: pair.upload_time, reverse=True)


def process_pair(
    pair: OptionFilePair,
    *,
    spot_price: float,
    iv_direction: str,
    output_directory: Path,
    processed_directory: Path,
    contract_multiplier: float = 100.0,
    create_charts: bool = False,
    enable_insights: bool = False,
    exclude_spx_columns: bool = False,
    debug_mode: bool = True,
) -> Dict[str, object]:
    """Combine and analyze ``pair``, then move the inputs into ``processed_directory``."""

    if spot_price is None:
        raise ValueError("spot_price must be provided when processing a file pair")

    output_directory = output_directory.expanduser().resolve()
    processed_directory = processed_directory.expanduser().resolve()

    output_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)

    if pair.version == "v2":
        from .v2_pipeline import ExposureRunConfig, run_exposure_pipeline

        run_config = ExposureRunConfig(
            ticker=pair.ticker,
            expiry=pair.expiry,
            spot=float(spot_price),
            contract_multiplier=contract_multiplier,
        )

        debug_dir = None
        if debug_mode:
            debug_dir = output_directory / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug mode enabled: writing debug exposures to %s", debug_dir)
        outputs = run_exposure_pipeline(
            pair.side_by_side_path,
            pair.greeks_path,
            run_config,
            output_dir=output_directory,
            debug_dir=debug_dir,
        )
        moved_files = [
            _move_to_processed(pair.side_by_side_path, processed_directory),
            _move_to_processed(pair.greeks_path, processed_directory),
        ]

        return {
            "pair": pair.to_dict(relative_to=pair.side_by_side_path.parent),
            "core_csv": str(outputs.core_path),
            "side_csv": str(outputs.side_path),
            "reactivity_csv": str(outputs.reactivity_path),
            "derived_csv": str(outputs.derived_path),
            "option_selling_csv": str(outputs.premium_path),
            "moved_files": [str(path) for path in moved_files],
            "iv_direction": iv_direction,
        }

    combined_df = combine_option_files(
        pair.side_by_side_path,
        pair.greeks_path,
        spot_price=spot_price,
        contract_multiplier=contract_multiplier,
    )

    combined_filename = _derive_combined_filename(pair)
    combined_path = output_directory / combined_filename
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(combined_path, index=False, header=COMBINED_CSV_HEADER)

    derived_dir = output_directory / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    calculation_time = datetime.now(timezone.utc)

    spx_exclusions = (
        {
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
        if exclude_spx_columns
        else None
    )
    derived_df = compute_derived_metrics(
        combined_df,
        calculation_time=calculation_time,
        spot_price=spot_price,
        iv_direction=iv_direction,
        drop_columns=spx_exclusions,
        include_totals_row=True,
    )
    market_state = derived_df.attrs.get("market_state")
    market_state_description = derived_df.attrs.get("market_state_description")
    market_state_components = derived_df.attrs.get("market_state_components")
    market_state_playbook = derived_df.attrs.get("market_state_playbook")
    vex_direction = derived_df.attrs.get("vex_direction")
    tex_direction = derived_df.attrs.get("tex_direction")
    gamma_box_high = derived_df.attrs.get("gamma_box_high")
    gamma_box_low = derived_df.attrs.get("gamma_box_low")
    breakout_up = derived_df.attrs.get("gamma_box_breakout_up")
    breakout_down = derived_df.attrs.get("gamma_box_breakout_down")
    vex_dir_box_high = derived_df.attrs.get("vex_dir_box_high")
    vex_dir_box_low = derived_df.attrs.get("vex_dir_box_low")
    tex_dir_box_high = derived_df.attrs.get("tex_dir_box_high")
    tex_dir_box_low = derived_df.attrs.get("tex_dir_box_low")
    safe_ticker = pair.ticker.replace("/", "-") or "unknown"
    safe_expiry = (
        pair.expiry.replace("/", "-") if pair.expiry != "UNKNOWN" else "unknown"
    )
    derived_filename = (
        f"derived_metrics_{safe_ticker}_{safe_expiry}_"
        f"{calculation_time.strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    derived_path = derived_dir / derived_filename
    formatted_derived_df = _format_derived_numeric_values(derived_df)
    formatted_derived_df.to_csv(derived_path, index=False)

    premium_path = _write_option_selling_components(
        combined_df,
        derived_dir,
        ticker=safe_ticker,
        expiry=safe_expiry,
        calculation_time=calculation_time,
    )

    insights_dir = derived_dir / "insights"
    insights_info: Dict[str, object] | None = None
    try:
        insights_info = generate_ai_insight(
            combined_df=combined_df,
            derived_df=derived_df,
            derived_path=derived_path,
            ticker=pair.ticker,
            expiry=pair.expiry,
            insights_dir=insights_dir,
        )
    except AIInsightsConfigurationError as exc:
        logger.warning("Skipping AI insight generation: %s", exc)
    except Exception:  # pragma: no cover - surfaced via CLI logging
        logger.exception("Failed to generate AI insight for %s", derived_path)

    analyzer_output_dir = output_directory / "analysis"
    analyzer_output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = BarchartOptionsAnalyzer(
        contract_multiplier=contract_multiplier,
        create_charts=create_charts,
        spot_price=float(spot_price),
        iv_direction=iv_direction,
        debug_mode=debug_mode,
    )

    results = analyzer.process_path(combined_path, analyzer_output_dir)
    summaries = _summaries_from_results(results)

    first_top5_detail = results[0].top5_detail if results else None

    market_structure_path = _write_market_structure_file(
        derived_path,
        market_state=market_state,
        market_state_description=market_state_description,
        market_state_components=market_state_components,
        market_state_playbook=market_state_playbook,
        derived_df=derived_df,
        ticker=pair.ticker,
        spot_price=spot_price,
        vex_direction=vex_direction,
        tex_direction=tex_direction,
        gamma_box_high=gamma_box_high,
        gamma_box_low=gamma_box_low,
        breakout_up=breakout_up,
        breakout_down=breakout_down,
        vex_dir_box_high=vex_dir_box_high,
        vex_dir_box_low=vex_dir_box_low,
        tex_dir_box_high=tex_dir_box_high,
        tex_dir_box_low=tex_dir_box_low,
        top5_detail=first_top5_detail,
    )

    moved_files = [
        _move_to_processed(pair.side_by_side_path, processed_directory),
        _move_to_processed(pair.greeks_path, processed_directory),
    ]

    return {
        "pair": pair.to_dict(relative_to=pair.side_by_side_path.parent),
        "combined_csv": str(combined_path),
        "moved_files": [str(path) for path in moved_files],
        "derived_csv": str(derived_path),
        "option_selling_csv": str(premium_path),
        "market_structure_txt": str(market_structure_path) if market_structure_path else None,
        "summaries": summaries,
        "insights": insights_info,
        "iv_direction": iv_direction,
        "market_state": market_state,
        "market_state_description": market_state_description,
        "market_state_components": market_state_components,
    }


def _summaries_from_results(results: Sequence[ProcessingResult]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for result in results:
        safe_suffix = f"{result.ticker}_{result.expiry}".replace("/", "-")
        summary_path = result.output_directory / f"{safe_suffix}_summary.json"
        per_strike_path = result.output_directory / f"{safe_suffix}_per_strike.csv"
        chart_path = result.output_directory / f"{safe_suffix}_charts.png"

        summary_info: Dict[str, object] = {
            "ticker": result.ticker,
            "expiry": result.expiry,
            "summary_json": str(summary_path),
            "per_strike_csv": str(per_strike_path),
        }

        if chart_path.exists():
            summary_info["chart"] = str(chart_path)

        summaries.append(summary_info)

    return summaries


def _move_to_processed(source: Path, destination_dir: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Source file {source} does not exist")

    destination = destination_dir / source.name
    if destination.exists():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        destination = destination_dir / f"{source.stem}_{timestamp}{source.suffix}"

    shutil.move(str(source), destination)
    return destination


def _derive_combined_filename(pair: OptionFilePair) -> str:
    expiry = pair.expiry.replace("/", "-") if pair.expiry != "UNKNOWN" else "unknown"
    ticker = pair.ticker.replace("/", "-") or "unknown"
    return f"unified_{ticker}_{expiry}.csv"


def _format_derived_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()

    def _format_value(value: object) -> object:
        if pd.isna(value):
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, Number):
            return format(float(value), ",") if isinstance(value, float) else format(value, ",")
        return value

    for column in formatted.columns:
        formatted[column] = formatted[column].apply(_format_value)

    return formatted


def _write_option_selling_components(
    combined_df: pd.DataFrame,
    output_dir: Path,
    *,
    ticker: str,
    expiry: str,
    calculation_time: datetime,
) -> Path:
    premium_source = combined_df.rename(
        columns={
            "Strike": "Strike",
            "net_gex": "Net_GEX",
            "net_dex": "Net_DEX",
            "dGEX/dSpot": "dGEX_dSpot",
            "dgex_dspot": "dGEX_dSpot",
            "call_theta": "Call_Theta",
            "call_open_interest": "Call_OI",
            "puts_theta": "Put_Theta",
            "puts_open_interest": "Put_OI",
        }
    )

    if "Net_DEX" not in premium_source:
        required_dex_columns = {
            "call_delta",
            "call_open_interest",
            "puts_delta",
            "puts_open_interest",
            "Spot",
        }
        if required_dex_columns.issubset(premium_source.columns):
            call_dex = (
                premium_source["call_delta"]
                * premium_source["Spot"]
                * premium_source["call_open_interest"]
                * 100.0
            )
            put_dex = (
                premium_source["puts_delta"]
                * premium_source["Spot"]
                * premium_source["puts_open_interest"]
                * 100.0
            )
            premium_source["Net_DEX"] = call_dex - put_dex
        else:
            premium_source["Net_DEX"] = 0.0

    for column in [
        "Net_GEX",
        "Net_DEX",
        "dGEX_dSpot",
        "Call_Theta",
        "Call_OI",
        "Put_Theta",
        "Put_OI",
    ]:
        if column not in premium_source:
            premium_source[column] = 0.0

    premium_df = build_premium_components(
        premium_source[
            [
                "Strike",
                "Net_GEX",
                "Net_DEX",
                "dGEX_dSpot",
                "Call_Theta",
                "Call_OI",
                "Put_Theta",
                "Put_OI",
            ]
        ]
    )

    premium_filename = (
        f"OptionSelling_premium_{ticker}_{expiry}_"
        f"{calculation_time.strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    premium_path = output_dir / premium_filename
    premium_df.to_csv(premium_path, index=False)
    return premium_path


def _write_market_structure_file(
    derived_path: Path,
    *,
    market_state: str | None,
    market_state_description: str | None,
    market_state_components: Dict[str, object] | None,
    market_state_playbook: Mapping[str, object] | None = None,
    derived_df: pd.DataFrame | None = None,
    ticker: str | None = None,
    spot_price: float | None = None,
    vex_direction: int | None = None,
    tex_direction: int | None = None,
    gamma_box_high: float | None = None,
    gamma_box_low: float | None = None,
    breakout_up: bool | None = None,
    breakout_down: bool | None = None,
    vex_dir_box_high: int | None = None,
    vex_dir_box_low: int | None = None,
    tex_dir_box_high: int | None = None,
    tex_dir_box_low: int | None = None,
    top5_detail: Mapping[str, object] | None = None,
) -> Path | None:
    """Persist the market structure summary next to the derived CSV.

    The returned path points to a ``*.txt`` file containing the market state
    name, its plain-English description, the individual classification
    components, the relevant playbook guidance, and the Gamma Box execution
    levels when available. If ``market_state`` is falsy, no file is written and
    ``None`` is returned.
    """

    if not market_state:
        return None

    target_path = derived_path.with_name(f"{derived_path.stem}_market_structure.txt")

    lines = [f"Market State: {market_state}"]

    if market_state_description:
        lines.append(f"Description: {market_state_description}")

    if market_state_components:
        lines.append("")
        lines.append("Components:")
        for key, value in sorted(market_state_components.items()):
            lines.append(f"- {key}: {value}")

    if market_state_playbook:
        next_step = market_state_playbook.get("next_step")
        useful_metrics = market_state_playbook.get("useful_metrics")
        avoid = market_state_playbook.get("avoid")

        if any([next_step, useful_metrics, avoid]):
            lines.append("")
            lines.append("Playbook:")

        if next_step:
            lines.append(f"Next Step: {next_step}")

        if useful_metrics:
            lines.append("Useful Metrics:")
            for metric in useful_metrics:
                lines.append(f"- {metric}")

        if avoid:
            lines.append(f"Avoid: {avoid}")

    def _interpret(direction: int | None, *, positive: str, negative: str, neutral: str) -> str:
        if direction is None:
            return "Unavailable"
        if direction > 0:
            return positive
        if direction < 0:
            return negative
        return neutral

    def _format_strike(value: float | int | str) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)

        if number.is_integer():
            return str(int(number))

        return f"{number:.2f}".rstrip("0").rstrip(".")

    def _format_box_direction(
        label: str,
        direction: int | None,
        *,
        positive: str,
        negative: str,
        neutral: str,
    ) -> str | None:
        if direction is None:
            return None

        return f"  {label}: {direction} (" + _interpret(
            direction,
            positive=positive,
            negative=negative,
            neutral=neutral,
        ) + ")"

    def _format_level(label: str, value: object) -> str:
        strike_value = "None" if value is None else _format_strike(value)
        return f"- {label}: {strike_value}"

    def _format_distance(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"

        if number.is_integer():
            return str(int(number))
        return f"{number:.2f}".rstrip("0").rstrip(".")

    def _calculate_magnets(
        df: pd.DataFrame | None,
        *,
        spot: float | None,
        ticker_symbol: str | None,
    ) -> tuple[list[float], int | None, float] | None:
        if df is None or spot is None:
            return None
        if "Strike" not in df.columns or "Net_GEX" not in df.columns:
            return None

        strikes = pd.to_numeric(df["Strike"], errors="coerce")
        net_gex = pd.to_numeric(df["Net_GEX"], errors="coerce")
        data = pd.DataFrame({"strike": strikes, "gex": net_gex}).dropna()
        if data.empty:
            return None

        magnitudes = data["gex"].abs()
        percentile_threshold = float(magnitudes.quantile(0.93)) if not magnitudes.empty else 0.0
        threshold = percentile_threshold

        candidates = data.loc[magnitudes >= threshold].copy()
        if candidates.empty:
            return None

        candidates["distance"] = (candidates["strike"] - float(spot)).abs()
        candidates = candidates.sort_values(["distance", "strike"])

        magnet_levels = [float(value) for value in candidates["strike"].tolist()[:10]]

        nearest_idx = data.assign(distance=(data["strike"] - float(spot)).abs())["distance"].idxmin()
        direction_value = data.loc[nearest_idx, "gex"]
        direction_sign: int | None
        if pd.isna(direction_value):
            direction_sign = None
        elif direction_value > 0:
            direction_sign = 1
        elif direction_value < 0:
            direction_sign = -1
        else:
            direction_sign = 0

        return magnet_levels, direction_sign, threshold

    if vex_direction is not None:
        lines.append("")
        lines.append("VEX Direction:")
        lines.append(f"- VEX_dir: {vex_direction}")
        lines.append(
            "- Interpretation: "
            + _interpret(
                vex_direction,
                positive="Upside fuel",
                negative="Downside fuel",
                neutral="No vol-based fuel",
            )
        )

    if tex_direction is not None:
        lines.append("")
        lines.append("TEX Direction:")
        lines.append(f"- TEX_dir: {tex_direction}")
        lines.append(
            "- Interpretation: "
            + _interpret(
                tex_direction,
                positive="Upward pressure",
                negative="Downward pressure",
                neutral="No theta-based pressure",
            )
        )

    execution_lines: List[str] = []
    magnets = _calculate_magnets(
        derived_df,
        spot=spot_price,
        ticker_symbol=ticker,
    )
    if gamma_box_high is not None:
        execution_lines.append(f"- Gamma_Box_High: {_format_strike(gamma_box_high)}")
        if breakout_up is not None:
            execution_lines.append(f"  Breakout_Up: {breakout_up}")
        vex_high_line = _format_box_direction(
            "VEX_dir_Box_high",
            vex_dir_box_high,
            positive="Upside fuel",
            negative="Downside fuel",
            neutral="Neutral",
        )
        if vex_high_line:
            execution_lines.append(vex_high_line)
        tex_high_line = _format_box_direction(
            "TEX_dir_Box_high",
            tex_dir_box_high,
            positive="Slow upside drift",
            negative="Slow downside drift",
            neutral="Neutral",
        )
        if tex_high_line:
            execution_lines.append(tex_high_line)
    if gamma_box_low is not None:
        execution_lines.append(f"- Gamma_Box_Low: {_format_strike(gamma_box_low)}")
        if breakout_down is not None:
            execution_lines.append(f"  Breakout_Down: {breakout_down}")
        vex_low_line = _format_box_direction(
            "VEX_dir_Box_low",
            vex_dir_box_low,
            positive="Upside fuel",
            negative="Downside fuel",
            neutral="Neutral",
        )
        if vex_low_line:
            execution_lines.append(vex_low_line)
        tex_low_line = _format_box_direction(
            "TEX_dir_Box_low",
            tex_dir_box_low,
            positive="Slow upside drift",
            negative="Slow downside drift",
            neutral="Neutral",
        )
        if tex_low_line:
            execution_lines.append(tex_low_line)

    if magnets:
        magnet_levels, direction_sign, threshold = magnets
        primary = magnet_levels[0]
        secondary = magnet_levels[1] if len(magnet_levels) > 1 else None
        execution_lines.append("- Magnets:")
        execution_lines.append(f"  Primary: {_format_strike(primary)}")
        if secondary is not None:
            execution_lines.append(f"  Secondary: {_format_strike(secondary)}")
        execution_lines.append(
            "  Levels (max 10): "
            + ", ".join(_format_strike(level) for level in magnet_levels)
        )
        if direction_sign is not None:
            execution_lines.append(
                "  Direction: "
                + f"{direction_sign} ("
                + _interpret(
                    direction_sign,
                    positive="Market pulled UP",
                    negative="Market pulled DOWN",
                    neutral="Balanced",
                )
                + ")"
            )
        execution_lines.append(f"  Threshold: {threshold:,.0f}")

    if execution_lines:
        lines.append("")
        lines.append("Execution:")
        lines.extend(execution_lines)

    if top5_detail:
        lines.append("")
        lines.append("Top 10 Detail:")

        lines.append("Levels:")
        lines.append(_format_level("Primary_Fade_Level", top5_detail.get("Primary_Fade_Level")))
        lines.append(
            _format_level(
                "Primary_Long_Drift_Level", top5_detail.get("Primary_Long_Drift_Level")
            )
        )
        lines.append(
            _format_level(
                "Primary_Short_Drift_Level", top5_detail.get("Primary_Short_Drift_Level")
            )
        )
        lines.append(_format_level("Flip_Zone", top5_detail.get("Flip_Zone")))
        lines.append(_format_level("Nearest_Magnet", top5_detail.get("Nearest_Magnet")))
        lines.append(_format_level("Secondary_Magnet", top5_detail.get("Secondary_Magnet")))

        detail_rows = top5_detail.get("Top5_Detail") if isinstance(top5_detail, Mapping) else None
        if detail_rows:
            lines.append("")
            lines.append("Top 10 Strikes:")
            for row in detail_rows:
                classification = row.get("Classification") or "Unclassified"
                regime = row.get("Regime") or "Unknown"
                energy = row.get("Energy_Score") or "Unknown"
                bias = row.get("Dealer_Bias") or "Unknown"
                distance = _format_distance(row.get("Distance_To_Spot"))
                strike = _format_strike(row.get("Strike", ""))
                lines.append(
                    f"- {strike}: {classification} | Regime: {regime} | Energy: {energy} | "
                    f"Dealer Bias: {bias} | Distance to Spot: {distance}"
                )

    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_path


__all__ = ["OptionFilePair", "discover_pairs", "process_pair"]
