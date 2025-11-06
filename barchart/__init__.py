"""Barchart Options Analyzer package."""

from importlib import import_module
from typing import Any

from .analyzer import BarchartOptionsAnalyzer, ProcessingResult, StrikeTypeMetric

__all__ = [
    "BarchartOptionsAnalyzer",
    "ProcessingResult",
    "StrikeTypeMetric",
    "combine_option_files",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import shim
    if name == "combine_option_files":
        module = import_module(".combiner", __name__)
        return module.combine_option_files
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
