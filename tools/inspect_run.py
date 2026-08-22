"""Look at the generations, grouped by what the run concluded about them.

Numbers can agree with each other and still be measuring a render that makes no sense.
This lays out every generation in a run as a source-over-perturbed strip, grouped into
vulnerabilities, no-effect and rejected, so the three groups can be compared by eye
against the verdicts the statistics produced.

Specifically worth looking for:

  CONSISTENCY   Does the edit hold across the clip, or appear and vanish? An edit that
                only exists for part of the window is a different intervention from the
                one the prompt describes.
  HALLUCINATION Objects nobody asked for. X2's characteristic failure is adding things -
                extra cups, a hand, a toy - and those are what a policy may actually be
                reacting to.
  NOTHING       A render judged "no effect" that is visually identical to its source
                tells you the editor declined, not that the policy is robust.

    .venv/Scripts/python tools/inspect_run.py runs/garage-ep0-... --group vulnerability
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402
from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402

TILE_W, TILE_H = 224, 126
LABEL_H = 20


def read_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else None


def group_of(s: dict) -> str:
    if s.get("is_vulnerability"):
        return "vulnerability"
    if s["status"] in ("rejected", "error"):
        return "rejected"
    if s["status"] == "done":
        return "no_effect"
    return "pending"


def label_bar(text: str, width: int, colour) -> np.ndarray:
    bar = np.full((LABEL_H, width, 3), 18, np.uint8)
    cv2.putText(bar, text[: width // 7], (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                colour, 1, cv2.LINE_AA)
    return bar


def strip(frames: np.ndarray, n: int) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return np.concatenate([cv2.resize(frames[i], (TILE_W, TILE_H)) for i in idx], axis=1)


COLOURS = {"vulnerability": (224, 160, 80), "no_effect": (111, 185, 126),
           "rejected": (222, 116, 119), "pending": (150, 150, 150)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--group", default="all",
                    choices=["all", "vulnerability", "no_effect", "rejected"])
    ap.add_argument("--frames", type=int, default=6, help="frames sampled per row")
    ap.add_argument("--max", type=int, default=8, help="scenarios per sheet")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    state = json.loads((args.run_dir / "state.json").read_text(encoding="utf-8"))
    picked = [s for s in state["scenarios"]
              if args.group == "all" or group_of(s) == args.group]
    picked = picked[: args.max]
    if not picked:
        raise SystemExit(f"no scenarios in group {args.group!r}")

    # The saved video is a grid: source row on top, perturbed below, cameras across.
    # One camera is enough to judge consistency, and keeps the sheet readable.
    rows = []
    for s in picked:
        video = args.run_dir / f"media/{s['name']}.mp4"
        frames = read_video(video) if video.exists() else None
        if frames is None:
            continue
        h = frames.shape[1] // 2
        w = frames.shape[2] // max(1, len(state.get("views") or [1]))
        source = frames[:, :h, :w]
        perturbed = frames[:, h:, :w]

        iv = s.get("intent") or {}
        c = s.get("vs_control") or {}
        tag = group_of(s).upper().replace("_", " ")
        detail = ""
        if c.get("p_value") is not None:
            detail = f"  p={c['p_value']:.4f} d={c.get('effect_size_cohens_d', 0):+.2f}"
        if iv.get("verdict") and iv["verdict"] != "unknown":
            detail += f"  | render: {iv['verdict']}"
        text = f"{tag}  {s['name']}{detail}"

        top, bot = strip(source, args.frames), strip(perturbed, args.frames)
        rows.append(label_bar(text, top.shape[1], COLOURS[group_of(s)]))
        rows.append(top)
        rows.append(bot)
        rows.append(np.full((6, top.shape[1], 3), 11, np.uint8))

    sheet = np.concatenate(rows, axis=0)
    out = args.out or (args.run_dir / f"inspect_{args.group}.jpg")
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"{len(rows) // 4} scenarios -> {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("each block: label, then SOURCE row, then PERTURBED row, left to right in time")


if __name__ == "__main__":
    main()
