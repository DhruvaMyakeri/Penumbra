"""Adjudicate what a finished run's renders actually show, and annotate it in place.

Runs from before the judge was wired into the garage have no `intent` field, and their
cards therefore assert a cause the run never checked. This reads the saved media,
adjudicates each render against its own prompt, and writes the verdict back into
`state.json` so the dashboard shows it.

**The media layout is not the same in every run, and getting it wrong silently ruins
every verdict.** Later runs save a *grid* - one column per perturbed camera, source row
above perturbed row - while the earliest runs saved a single perturbed camera at full
frame. Handing the grid to the judge as if it were one image produces verdicts about a
contact sheet ("the edited image is a 2x2 grid showing multiple views"), which look like
real adjudications and are worthless. The layout is detected from the run's own
`views` list and the tiles are cropped out before anything is judged.

Two limits remain, stated rather than buried:

  COMPRESSED   The media is re-encoded video, so the judge sees a compressed frame
               rather than exactly what the policy saw. Colour and structure survive
               that; fine detail may not, which biases toward "not applied".
  DOWNSCALED   Media tiles are smaller than the policy's input.

Neither applies to runs made after the judge was wired into the garage, which adjudicate
every perturbed camera on the frames the policy actually received.

    .venv/Scripts/python tools/judge_run.py runs/garage-ep0-...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402
from penumbra.episodes.types import PERTURBED_VIEW, VIEWS  # noqa: E402
from penumbra.scenarios.judge import judge_views  # noqa: E402


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


def split_media(stack: np.ndarray, views: list[str]) -> dict[str, np.ndarray]:
    """Recover each camera's PERTURBED tile from a saved media video.

    Two layouts exist. A grid is `len(views)` columns wide and two rows tall, source
    above perturbed. A single-view file is one camera at full frame, perturbed only.
    They are told apart by aspect: a grid of n cameras is about `2n` times wider than
    tall relative to a 16:9 tile, which no single view ever is.
    """
    h, w = stack.shape[1:3]
    n = max(1, len(views))
    tile_aspect = (w / n) / (h / 2)
    if n >= 1 and 1.4 < tile_aspect < 2.2 and h % 2 == 0:
        tw, th = w // n, h // 2
        return {v: stack[:, th:, i * tw:(i + 1) * tw] for i, v in enumerate(views)}
    # Single perturbed view, full frame.
    return {views[0] if views else PERTURBED_VIEW: stack}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--only-tested", action="store_true",
                    help="skip renders the gate rejected; nothing was claimed about them")
    args = ap.parse_args()

    state_path = args.run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    views = list(state.get("views") or [PERTURBED_VIEW])
    episode = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)

    print(f"cameras in media: {', '.join(views)}")
    print(f"{'situation':<32}{'outcome':<16}{'render judged'}")
    print("-" * 84)
    changed = 0
    for s in state["scenarios"]:
        if args.only_tested and s["status"] == "rejected":
            continue
        video = args.run_dir / f"media/{s['name']}.mp4"
        if not video.exists():
            continue
        stack = read_video(video)
        if stack is None:
            continue
        tiles = split_media(stack, views)
        n = min(min(len(t) for t in tiles.values()),
                min(len(episode.frames[v]) for v in tiles))
        source = {v: episode.frames[v][:n] for v in tiles}
        perturbed = {v: t[:n] for v, t in tiles.items()}

        v = judge_views(s["prompt"], s.get("situation", ""), source, perturbed,
                        list(tiles))
        d = v.to_dict()
        d["caveat"] = (d["caveat"] + " Retro-adjudicated from saved media tiles, not "
                                     "from the frames the policy received.")
        s["intent"] = d
        changed += 1
        outcome = ("VULNERABILITY" if s.get("is_vulnerability")
                   else ("rejected" if s["status"] == "rejected" else "no effect"))
        print(f"{s['name']:<32}{outcome:<16}{v.verdict}")
        if v.observed:
            print(f"{'':<48}{v.observed[:110]}")

    state["intent_audit"] = {
        "judged": changed,
        "cameras": views,
        "note": ("retro-adjudicated from saved media tiles; see each scenario's "
                 "`intent`. Adjudication is evidence about the render, not ground "
                 "truth, and nothing in the pipeline gates on it."),
    }
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    hits = [s for s in state["scenarios"] if s.get("is_vulnerability")]
    credible = [s for s in hits if (s.get("intent") or {}).get("credible")]
    print("-" * 84)
    print(f"judged {changed} renders across {len(views)} camera(s)")
    if hits:
        print(f"confirmed vulnerabilities whose NAME the render supports: "
              f"{len(credible)}/{len(hits)}")
        for s in hits:
            if s not in credible:
                iv = s.get("intent") or {}
                print(f"  ! {s['name']}: judged '{iv.get('verdict')}' - "
                      f"{iv.get('observed', '')[:90]}")
        print("\nA finding whose name is unsupported is still a real behaviour change. "
              "What it is not is evidence about the situation it is named after.")


if __name__ == "__main__":
    main()
