# Barnhart Options Analyzer

This repository provides a proof-of-concept implementation of the **Barnhart Options Analyzer**.
It ingests Barnhart-format options chain CSV files, computes aggregate Vanna and Gamma Exposure
(GEX) statistics, and produces summary artifacts for downstream research workflows.

## Features

- Parse one or many Barnhart CSV files and automatically infer ticker/expiry metadata from
  the filename convention (`$<ticker>-options-exp-YYYY-MM-DD-...csv`).
- Clean and normalize greeks, open interest, and price columns.
- Compute total and side-specific (calls vs puts) GEX and Vanna exposure.
- Export JSON summaries, per-strike CSV tables, and optional PNG curve charts for each input file.
- Command line interface for ad-hoc runs or batch processing directories.

## Getting Started

Create and activate a Python 3.11+ virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the analyzer against a Barnhart CSV file:

```bash
python -m barnhart.cli --input ./data/$spx-options-exp-2025-11-03.csv --out ./output/
```

An example dataset is included under `examples/` for quick smoke testing:

```bash
python -m barnhart.cli --input ./examples --out ./output --no-charts
```

Key options:

- `--input` (`-i`): CSV file or directory containing multiple CSVs.
- `--out` (`-o`): Destination directory for JSON/CSV/PNG outputs (created automatically).
- `--contract-multiplier`: Override the default contract size (100).
- `--no-charts`: Skip generating matplotlib PNG charts.
- `--log-level`: Adjust verbosity (e.g., `DEBUG`).

When a directory is supplied, every CSV within it is processed and individual summary artifacts are
written for each file.

## Output Artifacts

For an input file such as `$spx-options-exp-2025-11-03-weekly.csv`, the tool writes:

- `SPX_2025-11-03_summary.json`
- `SPX_2025-11-03_per_strike.csv`
- `SPX_2025-11-03_charts.png` (optional)

The JSON report includes aggregate GEX/Vanna totals and strike-level dictionaries that can be used
for dashboarding or quantitative modeling.

## Extensibility

This proof-of-concept is structured so that scheduling, data-source integrations, or alternative
reporting sinks (Slack, Streamlit dashboards, etc.) can be layered on top of the `BarnhartOptionsAnalyzer`
class in future iterations.
