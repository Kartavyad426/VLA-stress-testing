"""Serve docs/PRIORITIES_7Oct.html and persist its checkbox state to docs/priorities_state.json.

    python3 experiments/serve_priorities.py [--port 8765]
    then open http://localhost:8765/PRIORITIES_7Oct.html

Ticks are written to the JSON file (git-trackable, readable by any session). Opened as a plain
file instead, the page falls back to browser localStorage and offers an export button.
"""
import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
STATE = os.path.join(DOCS, "priorities_state.json")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=DOCS, **k)

    def do_GET(self):
        if self.path.split("?")[0] == "/priorities_state.json" and not os.path.exists(STATE):
            self._json(200, b"{}")
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/save":
            self._json(404, b'{"error":"not found"}')
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            state = json.loads(body)
            assert isinstance(state, dict)
        except (ValueError, AssertionError):
            self._json(400, b'{"error":"bad json"}')
            return
        tmp = STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, STATE)
        self._json(200, b'{"ok":true}')

    def _json(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"http://localhost:{a.port}/PRIORITIES_7Oct.html  (state: {os.path.normpath(STATE)})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
