"""A minimal forward proxy that sanitizes Codex requests on the way to vLLM.

Zero dependencies: stdlib only, so `pip install` pulls nothing and the adapter can run
anywhere Python 3.9+ runs, including inside the vLLM container itself.

Design notes:

* Only `/v1/responses` bodies are rewritten. Every other path -- `/v1/models`,
  `/v1/chat/completions`, `/metrics`, health -- is forwarded untouched, so the adapter
  is transparent to anything that is not Codex.
* Responses are streamed back as raw bytes and never parsed. SSE therefore works with
  no special handling, and a future vLLM that changes its event format cannot break the
  adapter.
* Errors from upstream are relayed with their original status and body, because
  swallowing a 400 into a 500 is exactly what makes this class of bug hard to diagnose.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .sanitize import sanitize_request
from .thinking import (
    DEFAULT_POLICY,
    PASSTHROUGH_POLICY,
    PolicyError,
    apply_thinking_policy,
    load_policy,
)

logger = logging.getLogger("codex-vllm-adapter")

# Hop-by-hop headers must not be forwarded (RFC 7230 6.1). Content-Length is dropped
# separately because sanitizing changes the body size.
_SKIP_REQUEST_HEADERS = {
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
    "accept-encoding",  # we do not decode, so ask upstream not to compress
}
_SKIP_RESPONSE_HEADERS = {
    "content-length", "connection", "keep-alive", "transfer-encoding", "upgrade",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = "http://127.0.0.1:8000"
    verbose = False
    extra_drop_types = frozenset()
    thinking_policy = DEFAULT_POLICY

    def log_message(self, fmt, *args):  # noqa: A003 - silence default stderr spam
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # BaseHTTPRequestHandler dispatches by method name.
    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_DELETE(self):
        self._forward("DELETE")

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _forward(self, method: str) -> None:
        body = self._read_body()

        if method == "POST" and self.path.rstrip("/").endswith("/responses") and body:
            body = self._sanitize(body)

        req = urllib.request.Request(
            self.upstream.rstrip("/") + self.path, data=body or None, method=method)
        for k, v in self.headers.items():
            if k.lower() not in _SKIP_REQUEST_HEADERS:
                req.add_header(k, v)
        if body:
            req.add_header("Content-Length", str(len(body)))

        try:
            resp = urllib.request.urlopen(req, timeout=None)
            status, headers, stream = resp.status, resp.headers, resp
        except urllib.error.HTTPError as e:
            # Relay upstream errors verbatim -- status AND body.
            status, headers, stream = e.code, e.headers, e
        except Exception as e:  # noqa: BLE001 - upstream unreachable
            logger.error("upstream %s unreachable: %s", self.upstream, e)
            payload = json.dumps({"error": {
                "message": "codex-vllm-adapter: upstream %s unreachable: %s"
                           % (self.upstream, e),
                "type": "upstream_unavailable"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(status)
        for k, v in headers.items():
            if k.lower() not in _SKIP_RESPONSE_HEADERS:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # Stream through in small chunks and flush each one, so SSE tokens reach Codex
        # as they are produced instead of buffering until the turn ends.
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("client disconnected mid-stream")
        finally:
            stream.close()

    def _sanitize(self, body: bytes) -> bytes:
        try:
            parsed = json.loads(body)
        except ValueError:
            # Not JSON: forward untouched rather than guessing.
            return body
        new, stats = sanitize_request(parsed, self.extra_drop_types)
        new, tstats = apply_thinking_policy(new, self.thinking_policy)
        stats.update(tstats)
        if not stats:
            return body
        if self.verbose:
            logger.info("sanitized %s: %s", self.path,
                        ", ".join("%s=%d" % kv for kv in sorted(stats.items())))
        return json.dumps(new).encode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="codex-vllm-adapter",
        description="Sanitizing proxy that makes stock vLLM accept Codex CLI traffic.")
    ap.add_argument("--listen", default="127.0.0.1:8010",
                    help="address to listen on (default: 127.0.0.1:8010)")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000",
                    help="vLLM base URL (default: http://127.0.0.1:8000)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="log what was removed from each request")
    ap.add_argument("--thinking", default=None, metavar="MODE",
                    help="thinking policy shorthand: 'off' (default) disables thinking "
                         "at every effort level; 'on' follows the client's effort, "
                         "clamped to a name Qwen's template accepts; 'keep' injects "
                         "nothing and lets vLLM decide. For per-level control use "
                         "--thinking-config.")
    ap.add_argument("--thinking-config", default=None, metavar="PATH",
                    help="JSON or TOML file mapping each reasoning effort to off / on / "
                         "keep / an effort name. See examples/thinking.toml.")
    ap.add_argument("--drop-item-type", action="append", default=[], metavar="TYPE",
                    help="also drop input items of this type (repeatable). Use when a "
                         "new item type causes a 500 whose traceback ends in "
                         "\"object has no attribute 'get'\" or \"KeyError: 'role'\".")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.thinking and args.thinking_config:
        ap.error("--thinking and --thinking-config are mutually exclusive")
    try:
        if args.thinking_config:
            policy = load_policy(args.thinking_config)
        elif args.thinking is None or args.thinking == "off":
            policy = DEFAULT_POLICY
        elif args.thinking == "on":
            policy = PASSTHROUGH_POLICY
        elif args.thinking == "keep":
            policy = None
        else:
            ap.error("--thinking must be one of: off, on, keep")
    except (PolicyError, OSError) as e:
        ap.error(str(e))

    host, _, port = args.listen.rpartition(":")
    Handler.upstream = args.upstream
    Handler.verbose = args.verbose
    Handler.extra_drop_types = frozenset(args.drop_item_type)
    Handler.thinking_policy = policy

    srv = ThreadingHTTPServer((host or "127.0.0.1", int(port)), Handler)
    srv.daemon_threads = True
    print("codex-vllm-adapter listening on %s -> %s" % (args.listen, args.upstream),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
