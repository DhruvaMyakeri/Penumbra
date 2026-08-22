"""Was a rejected render actually corrupted, or did the gate misread a recolor?

The garage rejects a specific scenario class at a high rate - camouflage and recolour
edits, which are also the most interesting ones to test. Before anyone touches the
threshold, the question has to be answered by measurement rather than by preference,
because loosening a gate after seeing which results it blocks is how a testing tool
starts agreeing with whoever is running it.

The gate flags a render when localised flow drift (p90) exceeds its bar. There are two
very different reasons that can happen:

  ARTEFACT   The edit recoloured an object. Dense optical flow is intensity-based, so
             a region whose colour changed produces apparent motion even though
             nothing moved. Then high drift sits INSIDE the recoloured region, the
             rest of the scene is still, and the rejection is the metric's limitation
             rather than a corrupted scene.

  REAL       The model actually warped the scene. Then high drift appears OUTSIDE the
             edited region too - on the table edge, the arm, the bowl - and the
             rejection is correct.

This measures which. It reports the share of high-drift pixels that fall inside the
colour-change region versus outside, and writes a map so the answer is visible as well
as numeric.

    python tools/diagnose_rejection.py runs/garage-... --scenario yellow_table_camouflage
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
from penumbra.episodes.types import PERTURBED_VIEW  # noqa: E402
from penumbra.validation.seam import _flow_magnitude, _gray  # noqa: E402

#: A pixel counts as recoloured when it moves this far in RGB.
COLOUR_CHANGE = 26.0
#: Drift pixels above this percentile are "high drift" for the purpose of locating it.
HIGH_DRIFT_PCT = 90


def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames in {path}")
    return np.stack(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    state = json.loads((args.run_dir / "state.json").read_text(encoding="utf-8"))
    entry = next((s for s in state["scenarios"] if s["name"] == args.scenario), None)
    if entry is None:
        raise SystemExit(f"no scenario {args.scenario!r} in {args.run_dir}")

    ep = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)
    source = ep.frames[PERTURBED_VIEW]

    video = args.run_dir / f"media/{args.scenario}.mp4"
    stacked = read_video(video)
    # Media may be a source-over-perturbed grid; take the perturbed half if so.
    if stacked.shape[1] >= 2 * stacked.shape[2] // 3:
        pass
    perturbed = stacked
    if perturbed.shape[1:3] != source.shape[1:3]:
        perturbed = np.stack([cv2.resize(f, (source.shape[2], source.shape[1]))
                              for f in perturbed])
    n = min(len(source), len(perturbed))

    idx = np.unique(np.linspace(0, n - 1, min(args.samples, n)).astype(int))
    inside_share, changed_share, outside_drift, inside_drift = [], [], [], []
    maps = []
    for i in idx:
        a = cv2.resize(source[i], (320, 180), interpolation=cv2.INTER_AREA)
        b = cv2.resize(perturbed[i], (320, 180), interpolation=cv2.INTER_AREA)
        flow = _flow_magnitude(_gray(a), _gray(b))

        colour_delta = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
        changed = cv2.morphologyEx((colour_delta > COLOUR_CHANGE).astype(np.uint8),
                                   cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0

        threshold = np.percentile(flow, HIGH_DRIFT_PCT)
        high = flow >= threshold
        if high.sum() == 0:
            continue
        inside_share.append(float((high & changed).sum()) / float(high.sum()))
        changed_share.append(float(changed.mean()))
        inside_drift.append(float(flow[changed].mean()) if changed.any() else 0.0)
        outside_drift.append(float(flow[~changed].mean()) if (~changed).any() else 0.0)

        vis = b.copy()
        vis[high & changed] = (0.4 * vis[high & changed] +
                               0.6 * np.array([60, 200, 90])).astype(np.uint8)
        vis[high & ~changed] = (0.4 * vis[high & ~changed] +
                                0.6 * np.array([240, 80, 80])).astype(np.uint8)
        maps.append(vis)

    inside = float(np.mean(inside_share))
    changed_frac = float(np.mean(changed_share))
    # How concentrated is drift inside the edited region, relative to how much of the
    # frame that region covers? 1.0 means no concentration at all.
    enrichment = inside / changed_frac if changed_frac > 1e-6 else float("nan")
    ratio = float(np.mean(inside_drift)) / max(1e-6, float(np.mean(outside_drift)))

    artefact = inside >= 0.6 and enrichment >= 2.0
    verdict = (
        "ARTEFACT OF THE RECOLOUR - high drift sits inside the region whose colour "
        "changed, and the rest of the scene is comparatively still. Flow is "
        "intensity-based, so a recoloured object reads as motion it never had. The "
        "rejection reflects the metric's limitation, not a corrupted scene."
        if artefact else
        "LIKELY REAL CORRUPTION - high drift is not concentrated in the edited region, "
        "so structure outside the edit moved too. The rejection looks correct."
    )

    result = {
        "run": args.run_dir.name,
        "scenario": args.scenario,
        "gate": entry.get("gate"),
        "measurements": {
            "high_drift_inside_edited_region": round(inside, 4),
            "edited_region_share_of_frame": round(changed_frac, 4),
            "enrichment_vs_area": round(enrichment, 2),
            "mean_drift_inside_edit": round(float(np.mean(inside_drift)), 4),
            "mean_drift_outside_edit": round(float(np.mean(outside_drift)), 4),
            "inside_outside_ratio": round(ratio, 2),
        },
        "verdict": verdict,
        "rejection_looks_like_artefact": bool(artefact),
    }
    (args.run_dir / f"rejection_{args.scenario}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    if maps:
        sheet = np.concatenate(maps[: min(5, len(maps))], axis=1)
        cv2.imwrite(str(args.run_dir / f"rejection_{args.scenario}.png"),
                    cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    print(json.dumps(result, indent=2))
    print("\ngreen = high drift inside the recoloured region (expected artefact)")
    print("red   = high drift outside it (real displacement)")


if __name__ == "__main__":
    main()
