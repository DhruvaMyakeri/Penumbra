"""Serve any completed (or in-flight) garage run's dashboard.

    python tools/view.py                       # newest run
    python tools/view.py runs/garage-ep0-...   # a specific one

Separate from `garage.py` so a finished run can be reopened without re-running it,
and so the viewer is never competing with a live run for the port.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penumbra.config import RUNS_DIR
from penumbra.ui.server import serve

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", nargs="?", type=Path)
ap.add_argument("--port", type=int, default=8766)
a = ap.parse_args()

run = a.run_dir
if run is None:
    runs = sorted(RUNS_DIR.glob("garage-*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("no garage runs found")
    run = runs[-1]
if not (run / "state.json").exists():
    raise SystemExit(f"{run} has no state.json")

serve(run, port=a.port)
n = len(list((run / "media").glob("*.jpg"))) if (run / "media").exists() else 0
print(f"  serving {run.name}  ({n} generations)")
print(f"  http://127.0.0.1:{a.port}\n  ctrl-c to stop")
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("stopped")
