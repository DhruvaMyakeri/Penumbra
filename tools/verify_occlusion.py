"""Did the occluder actually land where it was aimed?

The prompt-only `target_removal` fault taught this lesson expensively: X2 answered
"remove the cup" by making the cup translucent and leaving it in the gripper. The
behavioural null that followed was a statement about the prompt, not the policy.

So a spatial insertion gets checked before it is allowed to mean anything. Five
questions, each answered with a number rather than a glance:

  1. APPEARS      is there a substantial, contiguous change region at all?
  2. ON TARGET    is that region where the pointer aimed it?
  3. OVERLAPS     does it cover the pixels the task-relevant object occupied?
  4. PERSISTS     is it present and stable across frames, not flickering in and out?
  5. CONTAINED    is the rest of the scene - table, arm, bowl - left alone?

A run failing any of these is an intervention that did not land. Its policy result is
reported as UNINTERPRETABLE rather than as evidence about the policy.

    python tools/verify_occlusion.py runs/perturb-target_occlusion-<id>
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

#: A pixel counts as changed when it moves this far in RGB (0-255, per-channel mean).
CHANGE_THRESHOLD = 34.0
#: The change region must cover at least this fraction of the frame to count as an
#: inserted object rather than noise.
MIN_CHANGE_FRACTION = 0.012
#: How far the change centroid may sit from the pointer, in normalised frame units.
MAX_CENTROID_OFFSET = 0.22
#: Fraction of sampled frames in which the region must be present.
MIN_PERSISTENCE = 0.8
#: Outside the change region the scene must be essentially untouched.
MAX_OUTSIDE_CHANGE = 0.10

TARGET_HSV = ((20, 120, 120), (35, 255, 255))  # the yellow cup


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
        raise SystemExit(f"no frames decoded from {path}")
    return np.stack(frames)


def change_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Where did the picture substantially change? Cleaned of speckle."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    m = (diff > CHANGE_THRESHOLD).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return m.astype(bool)


def largest_blob(mask: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    """The biggest connected region, its area fraction, and its centroid."""
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 1:
        return np.zeros_like(mask), 0.0, (0.5, 0.5)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = labels == k
    h, w = mask.shape
    return blob, float(stats[k, cv2.CC_STAT_AREA]) / (h * w), (
        float(cent[k][0] / w), float(cent[k][1] / h)
    )


def target_mask(frame: np.ndarray) -> np.ndarray:
    lo, hi = TARGET_HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args()

    record = json.loads((args.run_dir / "experiment.json").read_text(encoding="utf-8"))
    # Pointers may be flat (one anchor) or keyed by view. Reading a per-view mapping
    # with .get("x") silently yields the 0.5 default and then measures the offset from
    # the middle of the frame, which is not where anything was aimed.
    raw = (record.get("perturbation") or {}).get("pointer") or {}
    if any(k in raw for k in ("x", "y", "active")):
        pointer = raw
    else:
        pointer = raw.get(PERTURBED_VIEW) or {}
    if not pointer:
        raise SystemExit(
            f"no pointer recorded for {PERTURBED_VIEW}; this fault has no spatial "
            f"anchor to verify against (recorded pointer: {raw})"
        )
    px, py = float(pointer["x"]), float(pointer["y"])

    ep = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)
    source = ep.frames[PERTURBED_VIEW]
    perturbed = read_video(args.run_dir / "perturbed.mp4")
    n = min(len(source), len(perturbed))
    source, perturbed = source[:n], perturbed[:n]
    if perturbed.shape[1:3] != source.shape[1:3]:
        perturbed = np.stack(
            [cv2.resize(f, (source.shape[2], source.shape[1])) for f in perturbed]
        )

    idx = np.unique(np.linspace(0, n - 1, min(args.samples, n)).astype(int))
    areas, centroids, overlaps, outside = [], [], [], []
    target_before, target_after = [], []

    for i in idx:
        m = change_mask(source[i], perturbed[i])
        blob, area, cent = largest_blob(m)
        areas.append(area)
        centroids.append(cent)

        tgt = target_mask(source[i])
        target_before.append(int(tgt.sum()))
        target_after.append(int(target_mask(perturbed[i]).sum()))
        overlaps.append(float((blob & tgt).sum()) / max(1.0, float(tgt.sum())))

        # How much of the scene changed *outside* the inserted region? A contained
        # insertion leaves the table, arm and bowl alone.
        rest = m & ~blob
        outside.append(float(rest.sum()) / rest.size)

    areas_a = np.asarray(areas)
    present = areas_a >= MIN_CHANGE_FRACTION
    persistence = float(present.mean())
    cents = np.asarray(centroids)[present] if present.any() else np.zeros((0, 2))
    centroid = cents.mean(axis=0) if len(cents) else np.array([np.nan, np.nan])
    centroid_offset = (
        float(np.hypot(centroid[0] - px, centroid[1] - py)) if len(cents) else float("inf")
    )
    centroid_jitter = float(np.hypot(*cents.std(axis=0))) if len(cents) > 1 else 0.0
    mean_overlap = float(np.mean(np.asarray(overlaps)[present])) if present.any() else 0.0
    mean_outside = float(np.mean(outside))
    tgt_before = float(np.mean(target_before))
    tgt_after = float(np.mean(target_after))

    checks = {
        "appears": bool(np.median(areas_a) >= MIN_CHANGE_FRACTION),
        "on_target": bool(centroid_offset <= MAX_CENTROID_OFFSET),
        "overlaps_target": bool(mean_overlap >= 0.25),
        "persists": bool(persistence >= MIN_PERSISTENCE),
        "contained": bool(mean_outside <= MAX_OUTSIDE_CHANGE),
    }
    gate = (record.get("divergence") or {}).get("evaluations", [{}])
    seam = (gate[0].get("validation") or {}).get("status") if gate else None
    checks["passes_seam"] = seam == "VALID"

    landed = all(checks.values())
    result = {
        "run": args.run_dir.name,
        "pointer": {"x": px, "y": py},
        "frames_compared": int(n),
        "measurements": {
            "median_change_area_fraction": round(float(np.median(areas_a)), 4),
            "change_centroid": [round(float(centroid[0]), 3), round(float(centroid[1]), 3)]
            if len(cents) else None,
            "centroid_offset_from_pointer": round(centroid_offset, 4)
            if np.isfinite(centroid_offset) else None,
            "centroid_jitter": round(centroid_jitter, 4),
            "mean_overlap_with_target": round(mean_overlap, 4),
            "persistence": round(persistence, 3),
            "mean_change_outside_region": round(mean_outside, 4),
            "target_pixels_source": round(tgt_before, 1),
            "target_pixels_perturbed": round(tgt_after, 1),
            "target_surviving_fraction": round(tgt_after / max(1.0, tgt_before), 4),
            "seam_status": seam,
        },
        "thresholds": {
            "min_change_fraction": MIN_CHANGE_FRACTION,
            "max_centroid_offset": MAX_CENTROID_OFFSET,
            "min_overlap": 0.25,
            "min_persistence": MIN_PERSISTENCE,
            "max_outside_change": MAX_OUTSIDE_CHANGE,
        },
        "checks": checks,
        "intervention_landed": landed,
        "interpretation": (
            "The occluder appeared where it was aimed, covered the target, persisted, "
            "and left the rest of the scene intact. A behavioural result from this run "
            "is about the policy."
            if landed else
            "The intervention did NOT land: "
            + ", ".join(f"{k} failed" for k, v in checks.items() if not v)
            + ". Any behavioural result from this run is UNINTERPRETABLE as evidence "
              "about the policy."
        ),
    }
    (args.run_dir / "occlusion_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    # Visual evidence: source / perturbed / change-region, five moments.
    show = np.unique(np.linspace(0, n - 1, 5).astype(int))
    rows = []
    for stack, label in ((source, "source"), (perturbed, "perturbed")):
        rows.append(np.concatenate([cv2.resize(stack[i], (256, 144)) for i in show], axis=1))
    overlay = []
    for i in show:
        m = change_mask(source[i], perturbed[i])
        blob, _, _ = largest_blob(m)
        vis = perturbed[i].copy()
        vis[blob] = (0.45 * vis[blob] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
        h, w = vis.shape[:2]
        cv2.drawMarker(vis, (int(px * w), int(py * h)), (0, 255, 255),
                       cv2.MARKER_CROSS, 34, 3)
        overlay.append(cv2.resize(vis, (256, 144)))
    rows.append(np.concatenate(overlay, axis=1))
    cv2.imwrite(str(args.run_dir / "occlusion_check.png"),
                cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))

    print(json.dumps(result, indent=2))
    print(f"\nrows: source / perturbed / change-region+pointer -> "
          f"{args.run_dir / 'occlusion_check.png'}")


if __name__ == "__main__":
    main()
