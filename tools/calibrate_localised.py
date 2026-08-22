"""Calibrate the gate against *localised* corruption - the case it was never tested on.

`tools/calibrate_gate.py` set both drift thresholds using **global** shifts. A global
shift moves every pixel equally, so its p90 equals its median: that ladder carries no
information at all about the failure mode p90 exists to catch, which is a *part* of the
scene moving while the rest stays put. The p90 bar in use (0.012) was therefore never
measured against the thing it gates.

This measures it, and separates two situations the current statistic cannot tell apart:

  RECOLOUR   A region changed colour and nothing moved. Farneback flow is intensity
             based, so the recoloured region reports apparent motion. p90 rises. The
             render is perfectly valid.

  DISPLACED  A region actually moved. p90 rises for a real reason.

The discriminator is drift measured **outside** the region whose colour changed. A
recolour leaves the rest of the frame still; a model that warped the scene does not.
This tool measures both statistics on synthetic corruptions of the real episode, where
ground truth is known by construction, plus the real X2 no-op render as a live control.

    .venv/Scripts/python tools/calibrate_localised.py

Blind to policy outcomes by construction: no policy is run and no scenario name appears.
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
from penumbra.episodes.types import PERTURBED_VIEW  # noqa: E402
from penumbra.validation.seam import _flow_magnitude, _gray  # noqa: E402

#: A pixel counts as recoloured when its mean RGB moves this far - the same bar the
#: rejection diagnosis uses, so the two tools speak about the same regions.
COLOUR_CHANGE = 26.0
WORK_W = 320


def _rect(shape, area):
    """A centred rectangle covering `area` of the frame."""
    h, w = shape
    side = float(np.sqrt(area))
    rh, rw = int(h * side), int(w * side)
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    return slice(y0, y0 + rh), slice(x0, x0 + rw)


def recolour(frames: np.ndarray, area: float) -> np.ndarray:
    """Change the colour of a region. Nothing moves. Must pass any honest gate."""
    out = frames.copy()
    ys, xs = _rect(frames.shape[1:3], area)
    region = out[:, ys, xs].astype(np.int16)
    # Rotate the channels and push saturation: a large, unambiguous colour change of
    # exactly the kind the scenario prompts ask for.
    out[:, ys, xs] = np.clip(region[..., [2, 0, 1]] * 1.15 + 20, 0, 255).astype(np.uint8)
    return out


def displace(frames: np.ndarray, area: float, pixels: int) -> np.ndarray:
    """Move a region by `pixels`, leaving the rest of the frame alone. Real corruption."""
    out = frames.copy()
    ys, xs = _rect(frames.shape[1:3], area)
    k = int(round(pixels * frames.shape[2] / 640.0))
    out[:, ys, xs] = np.roll(frames[:, ys, xs], k, axis=2)
    return out


def shift_all(frames: np.ndarray, pixels: int) -> np.ndarray:
    k = int(round(pixels * frames.shape[2] / 640.0))
    return np.roll(frames, k, axis=2)


def gamma(frames: np.ndarray, g: float) -> np.ndarray:
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return lut[frames]


def brighten(frames: np.ndarray) -> np.ndarray:
    return np.clip(frames.astype(np.float32) * 1.5 + 40, 0, 255).astype(np.uint8)


def measure(source: np.ndarray, perturbed: np.ndarray, samples: int = 12) -> dict:
    """Median drift, p90 drift, and drift restricted to pixels that did NOT recolour."""
    n = min(len(source), len(perturbed))
    idx = np.unique(np.linspace(0, n - 1, min(samples, n)).astype(int))
    med, p90, outside, area = [], [], [], []
    for i in idx:
        h, w = source.shape[1:3]
        th = int(round(h * WORK_W / w))
        a = cv2.resize(source[i], (WORK_W, th), interpolation=cv2.INTER_AREA)
        b = cv2.resize(perturbed[i], (WORK_W, th), interpolation=cv2.INTER_AREA)
        flow = _flow_magnitude(_gray(a), _gray(b))
        diag = float(np.hypot(*flow.shape))

        delta = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
        changed = cv2.morphologyEx((delta > COLOUR_CHANGE).astype(np.uint8),
                                   cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        med.append(float(np.median(flow)) / diag)
        p90.append(float(np.percentile(flow, 90)) / diag)
        area.append(float(changed.mean()))
        rest = flow[~changed]
        # If an edit covers the whole frame there is no "outside" to measure and the
        # statistic is undefined - report that rather than inventing a number.
        outside.append(float(np.percentile(rest, 90)) / diag if rest.size > 64 else float("nan"))
    return {
        "drift_median": round(float(np.median(med)), 5),
        "drift_p90": round(float(np.median(p90)), 5),
        "residual_p90": round(float(np.nanmedian(outside)), 5),
        "recoloured_area": round(float(np.mean(area)), 4),
    }


def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--x2-render", type=Path, default=None,
                    help="a real X2 no-op render (perturbed.mp4) to include as a live control")
    args = ap.parse_args()

    ep = load_episode([r for r in list_episodes() if r.episode_index == args.episode][0]
                      ).window(args.start, args.frames)
    src = ep.frames[PERTURBED_VIEW]

    valid = {
        "identity": measure(src, src),
        "gamma_0.5": measure(src, gamma(src, 0.5)),
        "brightness_1.5x+40": measure(src, brighten(src)),
    }
    for a in (0.05, 0.10, 0.30, 0.70):
        valid["recolour_%dpct_area" % int(a * 100)] = measure(src, recolour(src, a))

    if args.x2_render and args.x2_render.exists():
        rendered = read_video(args.x2_render)
        n = min(len(rendered), len(src))
        valid["x2_noop_render"] = measure(src[:n], rendered[:n])

    corrupt = {}
    for px in (2, 4, 8, 16):
        corrupt["shift_all_%dpx" % px] = measure(src, shift_all(src, px))
    for a in (0.05, 0.10, 0.30):
        for px in (4, 8, 16):
            corrupt["displace_%dpct_by_%dpx" % (int(a * 100), px)] = measure(src, displace(src, a, px))

    def col(d, k):
        return [v[k] for v in d.values() if not np.isnan(v[k])]

    worst_valid_p90 = max(col(valid, "drift_p90"))
    worst_valid_res = max(col(valid, "residual_p90"))
    best_bad_p90 = min(col(corrupt, "drift_p90"))
    best_bad_res = min(col(corrupt, "residual_p90"))

    result = {
        "appearance_only": valid,
        "geometry_moved": corrupt,
        "separation": {
            "worst_appearance_drift_p90": round(worst_valid_p90, 5),
            "smallest_corruption_drift_p90": round(best_bad_p90, 5),
            "drift_p90_separates": bool(worst_valid_p90 < best_bad_p90),
            "worst_appearance_residual_p90": round(worst_valid_res, 5),
            "smallest_corruption_residual_p90": round(best_bad_res, 5),
            "residual_p90_separates": bool(worst_valid_res < best_bad_res),
        },
        "recommended": {
            "max_residual_drift_p90": (
                round(float(np.sqrt(max(worst_valid_res, 1e-6) * best_bad_res)), 5)
                if worst_valid_res < best_bad_res else None),
        },
        "note": ("residual_p90 is the 90th-percentile flow over pixels whose colour did "
                 "NOT change. A recolour cannot raise it; a displacement can."),
    }
    out = RUNS_DIR / "_probe" / "gate_calibration_localised.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("%-30s%-12s%-12s%-14s%s" % ("transform", "drift med", "drift p90", "residual p90", "area"))
    print("-" * 78)
    for label, group in (("APPEARANCE ONLY - must pass", valid),
                         ("GEOMETRY MOVED - must fail", corrupt)):
        print("\n" + label)
        for k, v in group.items():
            print("  %-28s%-12.5f%-12.5f%-14.5f%.3f" % (
                k, v["drift_median"], v["drift_p90"], v["residual_p90"], v["recoloured_area"]))
    print("\n" + "-" * 78)
    s = result["separation"]
    print("drift_p90     valid<= %.5f   corrupt>= %.5f   separates: %s" % (
        s["worst_appearance_drift_p90"], s["smallest_corruption_drift_p90"],
        s["drift_p90_separates"]))
    print("residual_p90  valid<= %.5f   corrupt>= %.5f   separates: %s" % (
        s["worst_appearance_residual_p90"], s["smallest_corruption_residual_p90"],
        s["residual_p90_separates"]))
    print("\nwritten %s" % out)


if __name__ == "__main__":
    main()
