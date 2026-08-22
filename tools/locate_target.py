"""Locate the task-relevant object, so the PROVENANCE pointer is measured not guessed.

X2's set_pointer takes normalised frame coordinates. Rather than eyeballing where the
yellow cup is, segment it by hue over the evaluation window and report the centroid
with its spread. The numbers this prints are the ones in faults/target_removal.yaml.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penumbra.config import RUNS_DIR
from penumbra.episodes.droid import list_episodes, load_episode
from penumbra.episodes.types import PERTURBED_VIEW

PRESETS = {
    "yellow": ((20, 120, 120), (35, 255, 255)),
    "blue":   ((100, 120, 60), (130, 255, 255)),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--color", default="yellow", choices=sorted(PRESETS))
    ap.add_argument("--min-area", type=int, default=150)
    a = ap.parse_args()

    ep = load_episode([r for r in list_episodes() if r.episode_index == a.episode][0])
    ep = ep.window(a.start, a.frames)
    fr = ep.frames[PERTURBED_VIEW]
    H, W = fr.shape[1:3]
    lo, hi = PRESETS[a.color]
    hits = []
    for i in range(0, len(fr), 8):
        hsv = cv2.cvtColor(fr[i], cv2.COLOR_RGB2HSV)
        m = cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < a.min_area:
            continue
        hits.append({"frame": i, "x": float(cent[k][0] / W), "y": float(cent[k][1] / H),
                     "area_px": int(stats[k, cv2.CC_STAT_AREA])})
    if not hits:
        raise SystemExit(f"no {a.color} region above {a.min_area}px found")
    xs = np.array([h["x"] for h in hits]); ys = np.array([h["y"] for h in hits])
    rec = {
        "episode": ep.episode_id, "view": PERTURBED_VIEW, "color": a.color,
        "frames_sampled": len(hits),
        "pointer": {"x": round(float(xs.mean()), 3), "y": round(float(ys.mean()), 3)},
        "spread": {"x_std": round(float(xs.std()), 4), "y_std": round(float(ys.std()), 4)},
        "per_frame": hits,
    }
    out = RUNS_DIR / "_probe"; out.mkdir(parents=True, exist_ok=True)
    (out / f"target_{a.color}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "per_frame"}, indent=2))

main()
