# Barchart Options Analyzer

This repository provides a proof-of-concept implementation of the **Barchart Options Analyzer**.
It ingests Barchart-format options chain CSV files, computes aggregate Vanna and Gamma Exposure
(GEX) statistics, and produces summary artifacts for downstream research workflows.

## Features

- Parse one or many Barchart CSV files and automatically infer ticker/expiry metadata from
  the filename convention (`$<ticker>-options-exp-YYYY-MM-DD-...csv`).
- Clean and normalize greeks, open interest, and price columns.
- Compute total and side-specific (calls vs puts) GEX and Vanna exposure.
- Export JSON summaries, per-strike CSV tables, and optional PNG curve charts for each input file.
- Command line interface for ad-hoc runs or batch processing directories.

## Local Setup

1. **Install prerequisites**
   - Python 3.11 or newer (the project targets 3.11 specifically).
   - `pip` for dependency management.
   - (Optional) `virtualenv` or the built-in `venv` module for isolated environments.

2. **Clone the repository and enter the directory**

   ```bash
   git clone https://github.com/<your-org>/Stocktrading.git
   cd Stocktrading
   ```

3. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

4. **Install Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The requirements file installs the CLI, analyzer core, and optional web UI dependencies.

5. **(Optional) Upgrade matplotlib dependencies for improved chart rendering**

   If you plan to generate PNG charts, ensure your environment has a functioning Matplotlib backend. On
   headless Linux hosts you may also want to install system packages such as `libfreetype6` and `pkg-config`.

## Local Testing

After installation you can validate the setup with the included example dataset. The commands below will
process the sample CSV files and write outputs to `./output/`.

1. **Run the CLI against the provided examples**

   ```bash
   python -m barchart.cli \
     --input ./examples \
     --out ./output \
     --spot-price 5200 \
     --no-charts
   ```

   This will parse every CSV in `examples/`, compute aggregate exposure metrics, and emit JSON and CSV
   summaries for each file. The `--no-charts` flag skips PNG generation to keep the smoke test fast.

2. **Inspect the generated artifacts**

   The CLI writes results into `./output/`. You should see per-expiry JSON summaries and per-strike CSV files,
   e.g. `SPX_2025-11-03_summary.json` and `SPX_2025-11-03_per_strike.csv`.

3. **(Optional) Launch the web UI for manual workflows**

   ```bash
   python -m barchart.webserver --input-dir ./examples --output-dir ./output
   ```

   Navigate to <http://127.0.0.1:8000> to stage pairs of CSVs, supply a spot price, and trigger analysis runs
   from the browser. Processed files are automatically moved to an internal `processed/` directory to avoid
   duplicate work.

## Usage

Run the analyzer against a Barchart CSV file:

```bash
python -m barchart.cli --input ./data/$spx-options-exp-2025-11-03.csv --out ./output/ --spot-price 5200
```

An example dataset is included under `examples/` for quick smoke testing:

```bash
python -m barchart.cli --input ./examples --out ./output --spot-price 5200 --no-charts
```

Key options:

- `--input` (`-i`): CSV file or directory containing multiple CSVs.
- `--out` (`-o`): Destination directory for JSON/CSV/PNG outputs (created automatically).
- `--contract-multiplier`: Override the default contract size (100).
- `--spot-price`: Underlying spot price used for Vanna/GEX calculations (required).
- `--no-charts`: Skip generating matplotlib PNG charts.
- `--log-level`: Adjust verbosity (e.g., `DEBUG`).

When a directory is supplied, every CSV within it is processed and individual summary artifacts are
written for each file.

## Web UI

An interactive web interface is available for manually combining side-by-side and volatility/greeks
CSV exports, then running the analyzer once you are ready. Start the lightweight HTTP server with:

```bash
python -m barchart.webserver --input-dir ./examples --output-dir ./output
```

Open <http://127.0.0.1:8000> in your browser. The page lists every unprocessed pair of CSV files
and allows you to enter the spot price to use for calculations. After the pair is processed, the
source files are moved into an automatically managed `processed/` folder to keep the staging
directory tidy.

## Output Artifacts

For an input file such as `$spx-options-exp-2025-11-03-weekly.csv`, the tool writes:

- `SPX_2025-11-03_summary.json`
- `SPX_2025-11-03_per_strike.csv`
- `SPX_2025-11-03_charts.png` (optional)

The JSON report includes aggregate GEX/Vanna totals and strike-level dictionaries that can be used
for dashboarding or quantitative modeling.

## Extensibility

This proof-of-concept is structured so that scheduling, data-source integrations, or alternative
reporting sinks (Slack, Streamlit dashboards, etc.) can be layered on top of the `BarchartOptionsAnalyzer`
class in future iterations.
