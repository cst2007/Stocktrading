"""Utilities for generating a static HTML dashboard from persisted metrics."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class TickerSnapshot:
    """Aggregate analytics captured for a ticker at a specific timestamp."""

    calculated_at: str
    next_gex: float
    call_vanna: float
    put_vanna: float


def _fetch_tickers(connection: sqlite3.Connection) -> List[str]:
    """Return distinct tickers sorted alphabetically."""

    cursor = connection.execute(
        "SELECT DISTINCT ticker FROM option_metrics ORDER BY ticker COLLATE NOCASE;"
    )
    return [row[0] for row in cursor.fetchall()]


def _fetch_ticker_history(
    connection: sqlite3.Connection, ticker: str
) -> List[TickerSnapshot]:
    """Return up to the four most recent snapshots for ``ticker``."""

    cursor = connection.execute(
        """
        SELECT
            calculated_at,
            SUM(next_gex) AS next_gex,
            SUM(COALESCE(call_vanna, 0)) AS call_vanna,
            SUM(COALESCE(put_vanna, 0)) AS put_vanna
        FROM option_metrics
        WHERE ticker = ?
        GROUP BY calculated_at
        ORDER BY calculated_at DESC
        LIMIT 4;
        """,
        (ticker,),
    )

    snapshots: List[TickerSnapshot] = []
    for row in cursor.fetchall():
        snapshots.append(
            TickerSnapshot(
                calculated_at=str(row[0]),
                next_gex=float(row[1]) if row[1] is not None else 0.0,
                call_vanna=float(row[2]) if row[2] is not None else 0.0,
                put_vanna=float(row[3]) if row[3] is not None else 0.0,
            )
        )
    return snapshots


def _load_dashboard_data(database_path: Path) -> Dict[str, List[TickerSnapshot]]:
    """Load all data required for the dashboard from ``database_path``."""

    if not database_path.exists():
        raise FileNotFoundError(
            f"Database {database_path} does not exist. Did you run the analyzer first?"
        )

    with sqlite3.connect(database_path) as connection:
        tickers = _fetch_tickers(connection)
        return {ticker: _fetch_ticker_history(connection, ticker) for ticker in tickers}


def _render_html(data: Dict[str, Sequence[TickerSnapshot]]) -> str:
    """Render the interactive dashboard HTML."""

    serializable = {
        ticker: [snapshot.__dict__ for snapshot in snapshots]
        for ticker, snapshots in data.items()
    }
    payload = json.dumps(serializable, indent=2)

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Options Analytics Dashboard</title>
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        margin: 2rem;
        background-color: #f7f7f7;
        color: #1f2933;
      }}
      h1 {{
        font-size: 1.75rem;
        margin-bottom: 1rem;
      }}
      label {{
        font-weight: 600;
      }}
      select {{
        padding: 0.5rem 0.75rem;
        font-size: 1rem;
        margin-left: 0.5rem;
      }}
      table {{
        border-collapse: collapse;
        margin-top: 1.5rem;
        width: 100%;
        max-width: 800px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.1);
      }}
      th, td {{
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #d2d6dc;
        text-align: left;
      }}
      th {{
        background-color: #eef2f7;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }}
      tbody tr:last-child td {{
        border-bottom: none;
      }}
      .empty-state {{
        margin-top: 1.5rem;
        padding: 1rem 1.5rem;
        background-color: #fffbea;
        border: 1px solid #fcd34d;
        border-radius: 0.25rem;
        max-width: 600px;
      }}
      footer {{
        margin-top: 2rem;
        font-size: 0.85rem;
        color: #52606d;
      }}
    </style>
  </head>
  <body>
    <h1>Options Analytics Dashboard</h1>
    <div>
      <label for=\"ticker-select\">Select a ticker:</label>
      <select id=\"ticker-select\"></select>
    </div>
    <div id=\"content\"></div>
    <footer>
      Displaying up to the last four calculations for each ticker from the analytics database.
    </footer>
    <script>
      const DASHBOARD_DATA = {payload};

      const tickerSelect = document.getElementById('ticker-select');
      const content = document.getElementById('content');

      const tickers = Object.keys(DASHBOARD_DATA);

      function formatNumber(value) {{
        return Number(value).toLocaleString(undefined, {{ maximumFractionDigits: 2 }});
      }}

      function formatTimestamp(value) {{
        if (!value) {{
          return 'Unknown';
        }}
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {{
          return value;
        }}
        return date.toLocaleString();
      }}

      function renderTicker(ticker) {{
        const rows = DASHBOARD_DATA[ticker] || [];
        if (rows.length === 0) {{
          content.innerHTML = '<div class="empty-state">No calculations found for the selected ticker.</div>';
          return;
        }}

        const tableRows = rows
          .map((row) => `
            <tr>
              <td>${{formatTimestamp(row.calculated_at)}}</td>
              <td>${{formatNumber(row.next_gex)}}</td>
              <td>${{formatNumber(row.call_vanna)}}</td>
              <td>${{formatNumber(row.put_vanna)}}</td>
            </tr>
          `)
          .join('');

        content.innerHTML = `
          <table>
            <thead>
              <tr>
                <th>Time of Calculation</th>
                <th>Next GEX</th>
                <th>Call Vanna</th>
                <th>Put Vanna</th>
              </tr>
            </thead>
            <tbody>
              ${{tableRows}}
            </tbody>
          </table>
        `;
      }}

      function initialize() {{
        if (tickers.length === 0) {{
          content.innerHTML = '<div class="empty-state">The analytics database does not contain any tickers yet.</div>';
          return;
        }}

        tickerSelect.innerHTML = tickers
          .map((ticker) => `<option value="${{ticker}}">${{ticker}}</option>`)
          .join('');

        tickerSelect.addEventListener('change', (event) => {{
          renderTicker(event.target.value);
        }});

        renderTicker(tickers[0]);
      }}

      initialize();
    </script>
  </body>
</html>
"""


def generate_dashboard(database_path: Path, output_path: Path) -> Path:
    """Generate an HTML dashboard summarizing analytics for each ticker."""

    data = _load_dashboard_data(database_path)
    html = _render_html(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_cli(args: Iterable[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML dashboard from the analytics SQLite database.",
    )
    parser.add_argument(
        "--database",
        required=True,
        type=Path,
        help="Path to the option_metrics.sqlite database produced by the analyzer.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination HTML file for the generated dashboard.",
    )

    parsed = parser.parse_args(args=args)
    return generate_dashboard(parsed.database, parsed.output)


def main() -> None:  # pragma: no cover - CLI entry point helper
    run_cli()


if __name__ == "__main__":  # pragma: no cover - CLI entry point helper
    main()
