"""Phase 2 – AI powered insight generation for derived metrics."""

from __future__ import annotations

import json
import logging
import os
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
from openai import APIError, OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a professional options market strategist specializing in intraday 0DTE "
    "trading and dealer positioning."
)

PROMPT_TEMPLATE = textwrap.dedent(
    """
    Using the metrics below, provide detailed insight into how dealer hedging and gamma positioning may impact 0DTE trading dynamics.

    CONTEXT:
    - Date/Time: {timestamp}
    - Ticker: {ticker}
    - Expiration: {expiry}
    - This analysis is for intraday trades (focus on morning open and 1:30 PM dealer hedge adjustment period).

    DATA SNAPSHOT:
      Total Call Open Interest: {sum_call_OI}
      Total Put Open Interest: {sum_put_OI}
      Average Call Delta: {avg_call_Delta}
      Average Put Delta: {avg_put_Delta}
      Net Vanna: {net_Vanna}
      Net GEX: {net_GEX}
      Vanna/GEX Ratio: {Vanna_GEX_Ratio}
      Vanna/GEX Call Ratio: {Vanna_GEX_Call_Ratio}
      Strike Range: {min_strike} - {max_strike}

    QUESTIONS:
    1. Based on dealer hedging behavior, what is the likely morning trade setup (opening positioning, volatility bias, expected drift)?
    2. Around 1:30 PM ET, when dealers rebalance 0DTE hedges, how might this affect intraday gamma flows and price direction?
    3. Estimate probability (0–100%) of price movement toward or away from high GEX strike zones.
    4. Suggest a directional bias (bullish/bearish/neutral) for intraday 0DTE traders.
    5. Recommend an entry price range and target level(s) based on gamma/Vanna concentration zones.
    6. Identify any pinning strikes or acceleration risk zones.

    Output your response in this structured format:
    {{
      "timestamp": "{timestamp}",
      "ticker": "{ticker}",
      "expiry": "{expiry}",
      "morning_trade_view": "...",
      "afternoon_hedge_view": "...",
      "directional_bias": "...",
      "probability_estimate": "...",
      "entry_price_range": "...",
      "target_price_range": "...",
      "pinning_strikes": ["...", "..."],
      "summary": "..."
    }}
    """
)


EXPECTED_FIELDS = {
    "timestamp",
    "ticker",
    "expiry",
    "morning_trade_view",
    "afternoon_hedge_view",
    "directional_bias",
    "probability_estimate",
    "entry_price_range",
    "target_price_range",
    "pinning_strikes",
    "summary",
}


class AIInsightsConfigurationError(RuntimeError):
    """Raised when the OpenAI integration is misconfigured."""


@dataclass
class InsightMetrics:
    timestamp: str
    ticker: str
    expiry: str
    sum_call_OI: int
    sum_put_OI: int
    avg_call_Delta: float
    avg_put_Delta: float
    net_Vanna: float
    net_GEX: float
    Vanna_GEX_Ratio: float | None
    Vanna_GEX_Call_Ratio: float | None
    min_strike: float
    max_strike: float


def _format_timestamp(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def _ensure_float(value: float | int | None, *, decimals: int = 2) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return round(float(value), decimals)


def summarize_metrics(
    combined_df: pd.DataFrame,
    derived_df: pd.DataFrame,
    *,
    ticker: str,
    expiry: str,
) -> InsightMetrics:
    """Return aggregated metrics for prompt construction."""

    call_oi = int(float(combined_df["call_open_interest"].fillna(0).sum()))
    put_oi = int(float(combined_df["puts_open_interest"].fillna(0).sum()))

    avg_call_delta = _ensure_float(combined_df["call_delta"].mean())
    avg_put_delta = _ensure_float(combined_df["puts_delta"].mean())

    net_vanna = float(derived_df["net_Vanna"].fillna(0).sum())
    net_gex = float(derived_df["net_GEX"].fillna(0).sum())

    with pd.option_context("mode.use_inf_as_na", True):
        vanna_gex_ratio = net_vanna / net_gex if net_gex else None

        call_vanna_total = combined_df["call_vanna"].fillna(0).sum()
        call_gex_total = combined_df["call_gex"].fillna(0).sum()
        vanna_gex_call_ratio = (
            float(call_vanna_total) / float(call_gex_total)
            if call_gex_total
            else None
        )

    min_strike = float(combined_df["Strike"].min()) if not combined_df.empty else 0.0
    max_strike = float(combined_df["Strike"].max()) if not combined_df.empty else 0.0

    timestamp_value = derived_df["DateTime"].dropna().iloc[0] if "DateTime" in derived_df else None

    return InsightMetrics(
        timestamp=_format_timestamp(timestamp_value),
        ticker=ticker,
        expiry=expiry,
        sum_call_OI=call_oi,
        sum_put_OI=put_oi,
        avg_call_Delta=avg_call_delta,
        avg_put_Delta=avg_put_delta,
        net_Vanna=round(net_vanna, 2),
        net_GEX=round(net_gex, 2),
        Vanna_GEX_Ratio=round(vanna_gex_ratio, 2) if vanna_gex_ratio is not None else None,
        Vanna_GEX_Call_Ratio=
        round(vanna_gex_call_ratio, 2) if vanna_gex_call_ratio is not None else None,
        min_strike=round(min_strike, 2),
        max_strike=round(max_strike, 2),
    )


def build_prompt(metrics: InsightMetrics) -> str:
    payload = metrics.__dict__.copy()
    for key in ("Vanna_GEX_Ratio", "Vanna_GEX_Call_Ratio"):
        if payload[key] is None:
            payload[key] = "N/A"
    return PROMPT_TEMPLATE.format(**payload)


def _create_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AIInsightsConfigurationError(
            "OPENAI_API_KEY environment variable is required for AI insight generation"
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    organization = os.getenv("OPENAI_ORG")
    return OpenAI(api_key=api_key, base_url=base_url, organization=organization)


def _extract_message_content(response: Any) -> str:
    """Return the text body from a Chat Completions response."""

    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError):  # pragma: no cover - defensive guard
        return ""


def _parse_model_response(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Model response was empty")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            fragment = candidate[start : end + 1]
            return json.loads(fragment)
        raise


def _normalise_fields(parsed: Mapping[str, Any], fallback: InsightMetrics) -> Dict[str, Any]:
    normalised: Dict[str, Any] = {field: parsed.get(field) for field in EXPECTED_FIELDS}

    if not normalised["timestamp"]:
        normalised["timestamp"] = fallback.timestamp
    if not normalised["ticker"]:
        normalised["ticker"] = fallback.ticker
    if not normalised["expiry"]:
        normalised["expiry"] = fallback.expiry

    if not isinstance(normalised.get("pinning_strikes"), list):
        value = normalised.get("pinning_strikes")
        if isinstance(value, str):
            normalised["pinning_strikes"] = [segment.strip() for segment in value.split(",") if segment.strip()]
        else:
            normalised["pinning_strikes"] = []

    return normalised


def _write_log(log_path: Path, entry: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def generate_ai_insight(
    *,
    combined_df: pd.DataFrame,
    derived_df: pd.DataFrame,
    derived_path: Path,
    ticker: str,
    expiry: str,
    insights_dir: Path,
) -> Dict[str, Any]:
    """Send derived metrics to OpenAI and persist the structured response."""

    client = _create_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    metrics = summarize_metrics(combined_df, derived_df, ticker=ticker, expiry=expiry)
    prompt = build_prompt(metrics)

    insights_dir.mkdir(parents=True, exist_ok=True)
    stem = derived_path.stem
    prompt_path = insights_dir / f"{stem}_prompt.txt"
    response_path = insights_dir / f"{stem}_response.json"
    parsed_path = insights_dir / f"{stem}_insight.json"

    prompt_path.write_text(prompt, encoding="utf-8")

    logger.info("Submitting derived metrics to OpenAI model %s", model)
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except APIError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    body = _extract_message_content(response)
    response_path.write_text(body, encoding="utf-8")

    parsed = _parse_model_response(body)
    normalised = _normalise_fields(parsed, metrics)
    parsed_path.write_text(json.dumps(normalised, indent=2), encoding="utf-8")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "derived_file": str(derived_path),
        "prompt_file": str(prompt_path),
        "response_file": str(response_path),
        "insight_file": str(parsed_path),
        "model": model,
        "latency_ms": duration_ms,
    }
    _write_log(insights_dir / "insights.log", log_entry)

    logger.info(
        "OpenAI insight generated in %.2f ms; response saved to %s",
        duration_ms,
        parsed_path,
    )

    return {
        "model": model,
        "prompt": str(prompt_path),
        "response": str(response_path),
        "insight_json": str(parsed_path),
        "latency_ms": duration_ms,
        "data": normalised,
    }


__all__ = [
    "AIInsightsConfigurationError",
    "InsightMetrics",
    "build_prompt",
    "generate_ai_insight",
    "summarize_metrics",
]

