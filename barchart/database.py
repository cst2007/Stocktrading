"""SQLite persistence utilities for options analytics outputs."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .analyzer import StrikeTypeMetric


@dataclass
class _SerializableMetric:
    ticker: str
    strike: float
    option_type: str
    next_gex: float
    call_vanna: float | None
    put_vanna: float | None
    call_iv: float | None
    put_iv: float | None
    calculated_at: str


class AnalyticsDatabase:
    """Simple SQLite-backed sink for per-strike analytics metrics."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS option_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        strike REAL NOT NULL,
        option_type TEXT NOT NULL CHECK(option_type IN ('call', 'put')),
        next_gex REAL NOT NULL,
        call_vanna REAL,
        put_vanna REAL,
        call_iv REAL,
        put_iv REAL,
        calculated_at TEXT NOT NULL
    );
    """

    _INSERT = """
    INSERT INTO option_metrics (
        ticker, strike, option_type, next_gex, call_vanna, put_vanna, call_iv, put_iv, calculated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(self._DDL)
            connection.commit()

    def record_metrics(self, ticker: str, strike_metrics: Sequence["StrikeTypeMetric"]) -> None:
        if not strike_metrics:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        serializable = [
            self._convert_metric(ticker, metric, timestamp) for metric in strike_metrics
        ]

        with sqlite3.connect(self.database_path) as connection:
            connection.executemany(
                self._INSERT,
                [
                    (
                        metric.ticker,
                        metric.strike,
                        metric.option_type,
                        metric.next_gex,
                        metric.call_vanna,
                        metric.put_vanna,
                        metric.call_iv,
                        metric.put_iv,
                        metric.calculated_at,
                    )
                    for metric in serializable
                ],
            )
            connection.commit()

    def _convert_metric(
        self, ticker: str, metric: "StrikeTypeMetric", timestamp: str
    ) -> _SerializableMetric:
        option_type = metric.option_type.lower()
        if option_type not in {"call", "put"}:
            raise ValueError(f"Unsupported option type '{metric.option_type}'")

        call_vanna = metric.vanna if option_type == "call" else None
        put_vanna = metric.vanna if option_type == "put" else None
        call_iv = metric.iv if option_type == "call" else None
        put_iv = metric.iv if option_type == "put" else None

        return _SerializableMetric(
            ticker=ticker,
            strike=float(metric.strike),
            option_type=option_type,
            next_gex=float(metric.next_gex),
            call_vanna=self._normalize_optional(call_vanna),
            put_vanna=self._normalize_optional(put_vanna),
            call_iv=self._normalize_optional(call_iv),
            put_iv=self._normalize_optional(put_iv),
            calculated_at=timestamp,
        )

    @staticmethod
    def _normalize_optional(value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
