"""HTTP server that exposes a simple UI for processing option file pairs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Tuple

from .pair_processor import OptionFilePair, discover_pairs, process_pair


@dataclass(slots=True)
class ServerState:
    input_dir: Path
    output_dir: Path
    processed_dir: Path
    contract_multiplier: float
    create_charts: bool

    @property
    def static_dir(self) -> Path:
        return Path(__file__).resolve().parent / "web"


class PairProcessingRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, state: ServerState, **kwargs):
        self.state = state
        super().__init__(*args, directory=str(state.static_dir), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - inherited name
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()
        if self.path == "/api/pairs":
            self._handle_list_pairs()
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - inherited name
        if self.path == "/api/process":
            self._handle_process_request()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    # ------------------------------------------------------------------
    # API handlers
    # ------------------------------------------------------------------
    def _handle_list_pairs(self) -> None:
        try:
            pairs = discover_pairs(self.state.input_dir)
        except Exception as exc:  # pragma: no cover - surfaced via HTTP response
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        payload = {
            "pairs": [pair.to_dict(relative_to=self.state.input_dir) for pair in pairs]
        }
        self._send_json(payload)

    def _handle_process_request(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON payload: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        pair_id = payload.get("pair_id")
        if not pair_id:
            self._send_json({"error": "'pair_id' is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        if "spot_price" not in payload:
            self._send_json({"error": "'spot_price' is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            spot_price = float(payload["spot_price"])
        except (TypeError, ValueError):
            self._send_json({"error": "'spot_price' must be numeric"}, status=HTTPStatus.BAD_REQUEST)
            return

        pair_lookup: Dict[str, OptionFilePair] = {
            pair.key: pair for pair in discover_pairs(self.state.input_dir)
        }
        if pair_id not in pair_lookup:
            self._send_json({"error": f"Pair '{pair_id}' was not found"}, status=HTTPStatus.NOT_FOUND)
            return

        pair = pair_lookup[pair_id]
        try:
            result = process_pair(
                pair,
                spot_price=spot_price,
                output_directory=self.state.output_dir,
                processed_directory=self.state.processed_dir,
                contract_multiplier=self.state.contract_multiplier,
                create_charts=self.state.create_charts,
            )
        except Exception as exc:  # pragma: no cover - surfaced via HTTP response
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"status": "ok", "result": result})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def log_message(self, format: str, *args) -> None:  # noqa: A003 - match base class
        # Reduce logging noise by sending messages to stdout in a compact format.
        message = format % args
        print(f"[{self.log_date_time_string()}] {self.address_string()} {message}")

    def _send_json(self, payload: Dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a small web UI for processing Barchart CSV file pairs.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./examples"),
        help="Directory containing raw side-by-side and greeks CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory where combined CSVs and analyzer artifacts are written.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory where processed input CSVs are moved (defaults to <input-dir>/processed).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind the HTTP server to.")
    parser.add_argument("--port", type=int, default=8000, help="Port where the HTTP server listens.")
    parser.add_argument(
        "--contract-multiplier",
        type=float,
        default=100.0,
        help="Contract size used in exposure calculations (default: 100).",
    )
    parser.add_argument(
        "--enable-charts",
        action="store_true",
        help="Generate matplotlib charts when running the analyzer.",
    )
    return parser


def run_server(state: ServerState, *, host: str, port: int) -> Tuple[str, int]:
    handler = partial(PairProcessingRequestHandler, state=state)
    httpd = ThreadingHTTPServer((host, port), handler)

    bound_host, bound_port = httpd.server_address
    print(f"Serving pair processor on http://{bound_host}:{bound_port}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        print("Shutting down server...")
    finally:
        httpd.server_close()

    return bound_host, bound_port


def main() -> None:  # pragma: no cover - CLI entry point helper
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    processed_dir = (
        args.processed_dir.expanduser().resolve()
        if args.processed_dir is not None
        else input_dir / "processed"
    )

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    state = ServerState(
        input_dir=input_dir,
        output_dir=output_dir,
        processed_dir=processed_dir,
        contract_multiplier=args.contract_multiplier,
        create_charts=args.enable_charts,
    )

    run_server(state, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - CLI entry point helper
    main()
