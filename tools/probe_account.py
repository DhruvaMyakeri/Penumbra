"""Discover which models this API key may reach, without printing the key or token."""
from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penumbra.config import reactor_api_key  # noqa: E402

BASE = "https://api.reactor.inc"
key = reactor_api_key()


def get(path: str) -> tuple[int, str]:
    req = urllib.request.Request(BASE + path, headers={"Reactor-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


for path in ["/models", "/v1/models", "/me", "/account", "/keys", "/sessions", "/"]:
    code, body = get(path)
    print(f"GET {path} -> {code}: {body[:1200]}")
    print("-" * 60)

# Unscoped token: decode only its claims, never print the token itself.
from penumbra.config import reactor_api_key as _k  # noqa: E402
from reactor_sdk import fetch_jwt  # noqa: E402

try:
    tok = fetch_jwt(_k())
    payload = tok.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    claims = json.loads(base64.urlsafe_b64decode(payload))
    claims.pop("jti", None)
    print("UNSCOPED TOKEN CLAIMS:")
    print(json.dumps(claims, indent=2)[:4000])
except Exception as e:  # noqa: BLE001
    print("unscoped mint failed:", type(e).__name__, e)
