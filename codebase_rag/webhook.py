"""GitHub push webhook that re-indexes a repository when new commits land.

Run with `python -m codebase_rag webhook`, expose the port (e.g. with a tunnel) and add a
GitHub webhook (content type application/json, event: push) using GITHUB_WEBHOOK_SECRET.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from .config import Settings
from .embeddings import get_embedder
from .pipeline import index_github_repo


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    """Check GitHub's X-Hub-Signature-256 header in constant time."""
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)


def repo_url_from_push(payload: dict) -> str | None:
    """Clone URL of the repo in a push event, or None if it is not a default-branch push."""
    repo = payload.get("repository") or {}
    default_ref = f"refs/heads/{repo.get('default_branch', 'main')}"
    if payload.get("ref") != default_ref:
        return None
    return repo.get("html_url")


def make_handler(settings: Settings, embedder_name: str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (stdlib naming)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not verify_signature(settings.webhook_secret, body, self.headers.get("X-Hub-Signature-256")):
                return self._reply(401, "invalid signature")
            if self.headers.get("X-GitHub-Event") != "push":
                return self._reply(200, "ignored event")
            url = repo_url_from_push(json.loads(body))
            if not url:
                return self._reply(200, "ignored ref")
            Thread(
                target=lambda: index_github_repo(url, get_embedder(embedder_name), settings), daemon=True
            ).start()
            self._reply(202, f"re-indexing {url}")

        def _reply(self, status: int, message: str):
            self.send_response(status)
            self.end_headers()
            self.wfile.write(message.encode())

        def log_message(self, *args):  # keep the console quiet
            pass

    return Handler


def serve(settings: Settings, embedder_name: str, port: int = 8765) -> None:
    if not settings.webhook_secret:
        raise SystemExit("Set GITHUB_WEBHOOK_SECRET before starting the webhook server.")
    print(f"Listening for GitHub push events on :{port}")
    HTTPServer(("", port), make_handler(settings, embedder_name)).serve_forever()
