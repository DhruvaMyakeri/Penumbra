"""Make an existing run's videos playable in a browser.

Every run written before this session used OpenCV's `mp4v` fourcc, which is MPEG-4
Part 2 — a codec no current browser will decode. The files are valid and the server
serves them; the `<video>` element just stays black. This re-encodes them in place to
H.264, which every browser plays, and skips anything already playable.

    .venv/Scripts/python tools/transcode.py runs/garage-ep0-...     # one run
    .venv/Scripts/python tools/transcode.py --all                   # every run

The originals are replaced. That is deliberate: two copies of the same generation with
different codecs is a way to end up looking at the wrong one. The frames are unchanged
— only the container and codec differ — so nothing about any recorded result moves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402
from penumbra.media import is_browser_playable, write_h264  # noqa: E402


def read_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, float(fps)


def transcode(path: Path) -> str:
    if is_browser_playable(path):
        return "already playable"
    frames, fps = read_frames(path)
    if not frames:
        return "unreadable"
    tmp = path.with_suffix(".h264.mp4")
    write_h264(tmp, frames, fps)
    # Replace only once the new file exists and decodes, so a failure mid-encode
    # cannot leave a run with no video at all.
    if not is_browser_playable(tmp):
        tmp.unlink(missing_ok=True)
        return "FAILED to encode"
    path.unlink()
    tmp.rename(path)
    return f"{len(frames)} frames -> h264"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--all", action="store_true", help="every run under runs/")
    args = ap.parse_args()

    if args.all:
        runs = sorted(p.parent for p in RUNS_DIR.glob("*/media"))
    elif args.run_dir:
        runs = [args.run_dir]
    else:
        raise SystemExit("give a run directory or --all")

    total = converted = 0
    for run in runs:
        videos = sorted((run / "media").glob("*.mp4")) if (run / "media").exists() else []
        videos += sorted(run.glob("*.mp4"))
        if not videos:
            continue
        print(f"\n{run.name}")
        for v in videos:
            result = transcode(v)
            total += 1
            converted += "h264" in result
            print(f"  {v.name:<40}{result}")
    print(f"\n{converted} of {total} videos re-encoded")


if __name__ == "__main__":
    main()
