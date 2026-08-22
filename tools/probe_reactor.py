"""Reactor capability probe.

Connects to one or more Reactor models, records every status transition, the
capabilities announcement, the declared tracks, and the full command schema.
Writes a JSON dossier per model to runs/_probe/.

Nothing here is inferred: whatever the server sends is what gets written.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR, reactor_api_key  # noqa: E402
from reactor_sdk import Reactor  # noqa: E402

OUT = RUNS_DIR / "_probe"


async def probe(model: str, timeout: float = 120.0) -> dict:
    key = reactor_api_key()
    record: dict = {
        "model": model,
        "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status_timeline": [],
        "capabilities": None,
        "messages": [],
        "errors": [],
        "tracks": None,
        "schema": None,
        "connect_seconds": None,
        "outcome": "unknown",
    }
    t0 = time.time()
    r = Reactor(model, api_key=key)

    ready = asyncio.get_running_loop().create_future()

    @r.on_status
    def _status(status: str) -> None:
        record["status_timeline"].append({"t": round(time.time() - t0, 3), "status": getattr(status, "value", str(status))})
        if getattr(status, "value", str(status)) == "ready" and not ready.done():
            ready.set_result(True)

    r.on("capabilities_received", lambda caps: record.__setitem__("capabilities", caps))
    r.on("message", lambda m: record["messages"].append(m))
    r.on("error", lambda e: record["errors"].append(f"{type(e).__name__}: {e}"))

    try:
        await asyncio.wait_for(r.connect(), timeout=timeout)
        await asyncio.wait_for(ready, timeout=timeout)
        record["connect_seconds"] = round(time.time() - t0, 3)
        record["session_id"] = r.session_id
        record["tracks"] = [
            {"name": t.name, "kind": getattr(t.kind, "value", str(t.kind)), "direction": getattr(t.direction, "value", str(t.direction))} for t in r.tracks
        ]
        try:
            record["schema"] = await asyncio.wait_for(r.request_schema(), timeout=60)
        except Exception as exc:  # noqa: BLE001
            record["schema_error"] = f"{type(exc).__name__}: {exc}"
        record["outcome"] = "ok"
    except Exception as exc:  # noqa: BLE001
        record["outcome"] = "failed"
        record["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            await asyncio.wait_for(r.disconnect(), timeout=30)
        except Exception:  # noqa: BLE001
            pass
        r.close()
    return record


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["reactor/x2"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.models or ["reactor/x2"]:
        print(f"=== probing {model} ===", flush=True)
        rec = await probe(model)
        path = OUT / f"{model.replace('/', '_')}.json"
        path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        print(f"outcome={rec['outcome']} connect={rec.get('connect_seconds')}s -> {path}")
        if rec["outcome"] != "ok":
            print("  failure:", rec.get("failure"))
        else:
            print("  tracks:", rec["tracks"])
            caps = rec.get("capabilities") or {}
            if isinstance(caps, dict):
                print("  capability keys:", list(caps.keys()))


if __name__ == "__main__":
    asyncio.run(main())
