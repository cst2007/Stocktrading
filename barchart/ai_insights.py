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

try:  # pragma: no cover - exercised indirectly via configuration
    from openai import APIError, OpenAI
except ModuleNotFoundError as import_error:  # pragma: no cover - import guard
    APIError = RuntimeError  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = import_error
else:
    _OPENAI_IMPORT_ERROR = None

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
      Total Net DEX: {net_DEX_total}
      Call Vanna Ratio: {Call_Vanna_Ratio}
      Put Vanna Ratio: {Put_Vanna_Ratio}
      Net Vanna/GEX Ratio: {Vanna_GEX_Total}
      Energy Score: {Energy_Score}
      Dealer Bias: {Dealer_Bias}
      IV Direction: {IV_Direction}
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
    net_DEX_total: float
    Call_Vanna_Ratio: float | None
    Put_Vanna_Ratio: float | None
    Vanna_GEX_Total: float | None
    Energy_Score: str
    Dealer_Bias: str
    IV_Direction: str
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

    if "Strike" in derived_df.columns:
        base_derived = derived_df.loc[
            derived_df["Strike"].astype(str).str.lower() != "total"
        ]
    else:
        base_derived = derived_df

    def _sum_numeric(series: pd.Series | None) -> float:
        if series is None:
            return 0.0
        return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())

    call_oi = int(float(combined_df["call_open_interest"].fillna(0).sum()))
    put_oi = int(float(combined_df["puts_open_interest"].fillna(0).sum()))

    avg_call_delta = _ensure_float(combined_df["call_delta"].mean())
    avg_put_delta = _ensure_float(combined_df["puts_delta"].mean())

    net_vanna = _sum_numeric(base_derived.get("net_Vanna"))
    net_gex = _sum_numeric(base_derived.get("net_GEX"))
    if "net_DEX" in base_derived:
        net_dex_total = _sum_numeric(base_derived["net_DEX"])
    else:
        net_dex_total = float(derived_df.attrs.get("total_net_DEX", 0.0))

    with pd.option_context("mode.use_inf_as_na", True):
        call_vanna_total = combined_df["call_vanna"].fillna(0).sum()
        call_gex_total = combined_df["call_gex"].fillna(0).sum()
        call_vanna_ratio = float(call_vanna_total) / float(call_gex_total) if call_gex_total else None

        put_vanna_total = combined_df["puts_vanna"].fillna(0).sum()
        put_gex_total = combined_df["puts_gex"].fillna(0).sum()
        put_vanna_ratio = float(put_vanna_total) / float(put_gex_total) if put_gex_total else None

        vanna_gex_total = net_vanna / net_gex if net_gex else None

    iv_direction = ""
    if "IV_Direction" in base_derived.columns:
        direction_series = base_derived["IV_Direction"].dropna().astype(str)
        if not direction_series.empty:
            iv_direction = direction_series.iloc[0]
    if not iv_direction:
        iv_direction = str(base_derived.attrs.get("iv_direction", ""))

    energy_score = ""
    dealer_bias = ""
    if {"IVxOI", "Energy_Score", "Dealer_Bias"}.issubset(base_derived.columns):
        ivxoi_series = pd.to_numeric(base_derived["IVxOI"], errors="coerce")
        if not ivxoi_series.dropna().empty:
            top_index = ivxoi_series.idxmax()
            energy_score = str(base_derived.at[top_index, "Energy_Score"])
            dealer_bias = str(base_derived.at[top_index, "Dealer_Bias"])

    min_strike = float(combined_df["Strike"].min()) if not combined_df.empty else 0.0
    max_strike = float(combined_df["Strike"].max()) if not combined_df.empty else 0.0

    timestamp_value = None
    if "DateTime" in base_derived.columns and not base_derived["DateTime"].dropna().empty:
        timestamp_value = base_derived["DateTime"].dropna().iloc[0]
    elif "DateTime" in derived_df and not derived_df["DateTime"].dropna().empty:
        timestamp_value = derived_df["DateTime"].dropna().iloc[0]

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
        net_DEX_total=round(net_dex_total, 2),
        Call_Vanna_Ratio=round(call_vanna_ratio, 2) if call_vanna_ratio is not None else None,
        Put_Vanna_Ratio=round(put_vanna_ratio, 2) if put_vanna_ratio is not None else None,
        Vanna_GEX_Total=round(vanna_gex_total, 2) if vanna_gex_total is not None else None,
        Energy_Score=energy_score,
        Dealer_Bias=dealer_bias,
        IV_Direction=iv_direction,
        min_strike=round(min_strike, 2),
        max_strike=round(max_strike, 2),
    )


def build_prompt(metrics: InsightMetrics) -> str:
    payload = metrics.__dict__.copy()
    for key in ("Call_Vanna_Ratio", "Put_Vanna_Ratio", "Vanna_GEX_Total"):
        if payload[key] is None:
            payload[key] = "N/A"
    if not payload["Energy_Score"]:
        payload["Energy_Score"] = "Unknown"
    if not payload["Dealer_Bias"]:
        payload["Dealer_Bias"] = "Neutral"
    if not payload["IV_Direction"]:
        payload["IV_Direction"] = "Unknown"
    return PROMPT_TEMPLATE.format(**payload)


def _create_client() -> OpenAI:
    if OpenAI is None:
        raise AIInsightsConfigurationError(
            "The optional 'openai' package is not installed; install it or disable AI insights"
        ) from _OPENAI_IMPORT_ERROR

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
        message = response.choices[0].message
    except (AttributeError, IndexError, KeyError):  # pragma: no cover - defensive guard
        return ""

    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content

    parts: list[str] = []

    if isinstance(content, (list, tuple)):
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue

            text = None
            if hasattr(part, "text"):
                text = getattr(part, "text")
            elif isinstance(part, Mapping):
                candidate = part.get("text")
                if candidate is not None:
                    text = candidate

            if text:
                parts.append(str(text))

    elif isinstance(message, Mapping):
        value = message.get("content")
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            for part in value:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, Mapping) and part.get("text"):
                    parts.append(str(part["text"]))

    if parts:
        return "\n".join(part.strip() for part in parts if part.strip())

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

