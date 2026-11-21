# SPDX-License-Identifier: MulanPSL-2.0
"""Debug Web UI server for visual_object (default http://127.0.0.1:8094)."""
from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

log = logging.getLogger("visual_object_rbnx.webui")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
MAX_POST_BYTES = 256 * 1024


class VisualObjectWebUI:
    def __init__(
        self,
        host: str,
        port: int,
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        preview_jpeg_fn: Callable[[], bytes | None],
    ) -> None:
        self._host = host
        self._port = int(port)
        self._handlers = dict(handlers)
        self._preview_jpeg_fn = preview_jpeg_fn
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is not None:
            return int(self._server.server_address[1])
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def start(self) -> None:
        if self._server is not None:
            return
        handlers = self._handlers
        preview_fn = self._preview_jpeg_fn

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_bytes(self, status: HTTPStatus, ctype: str, body: bytes) -> None:
                self.send_response(int(status))
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self._send_bytes(status, "application/json; charset=utf-8", body)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_POST_BYTES:
                    raise ValueError("request body too large")
                if length == 0:
                    return {}
                raw = self.rfile.read(length)
                data = json.loads(raw.decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("JSON body must be an object")
                return data

            def do_HEAD(self) -> None:  # noqa: N802
                self.do_GET()

            def do_GET(self) -> None:  # noqa: N802
                route = urlsplit(self.path).path.strip("/")
                if route in ("", "index.html"):
                    # Merged into 8092 Local Mapping WebUI; redirect browser visits cleanly.
                    self.send_response(int(HTTPStatus.FOUND))
                    self.send_header("Location", "http://127.0.0.1:8092/#visual_object")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if route == "api/preview.jpg":
                    jpeg = preview_fn()
                    if not jpeg:
                        self._send_json(HTTPStatus.NO_CONTENT, {"ok": False, "detail": "no camera frame yet"})
                        return
                    self._send_bytes(HTTPStatus.OK, "image/jpeg", jpeg)
                    return
                fn = handlers.get(f"GET:{route}")
                if fn is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "detail": "not found"})
                    return
                try:
                    self._send_json(HTTPStatus.OK, fn({}))
                except Exception as exc:  # noqa: BLE001
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "detail": str(exc)})

            def do_POST(self) -> None:  # noqa: N802
                route = urlsplit(self.path).path.strip("/")
                fn = handlers.get(f"POST:{route}")
                if fn is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "detail": "unsupported endpoint"})
                    return
                try:
                    body = self._read_json()
                except Exception as exc:  # noqa: BLE001
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "detail": str(exc)})
                    return
                try:
                    res = fn(body)
                except Exception as exc:  # noqa: BLE001
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "detail": str(exc)})
                    return
                status = HTTPStatus.OK if res.get("ok", True) else HTTPStatus.CONFLICT
                self._send_json(status, res)

        class _ReusableHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = _ReusableHTTPServer((self._host, self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.25},
            name="visual-object-webui",
            daemon=True,
        )
        self._thread.start()
        log.info("visual_object debug web UI listening on %s", self.url)

    def stop(self) -> None:
        srv, self._server = self._server, None
        thr, self._thread = self._thread, None
        if srv is not None:
            try:
                srv.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                srv.server_close()
            except Exception:  # noqa: BLE001
                pass
        if thr is not None:
            thr.join(timeout=2.0)
