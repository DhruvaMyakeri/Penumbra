"""Did the PROVENANCE intervention actually happen?

A null result from object removal — "behaviour did not change" — is only interpretable
if the object was really removed. If X2 quietly ignored the prompt, the null is about
the prompt, not the policy, and reporting it as a finding about the policy would be
exactly the kind of fabrication this project forbids.

So the intervention gets its own measurement, independent of the policy: count the
target-coloured pixels in the source stream and in the perturbed stream, inside and
outside the pointer region. Reports the fraction of the target that survived.

    python tools/verify_removal.py runs/perturb-target_removal-YYYYMMDD-HHMMSS
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

PRESETS = {"yellow": ((20, 120, 120), (35, 255, 255))}


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


def target_mask(frames: np.ndarray, color: str) -> np.ndarray:
    lo, hi = PRESETS[color]
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        m = cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        out.append(m > 0)
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--color", default="yellow", choices=sorted(PRESETS))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--radius", type=float, default=0.14,
                    help="pointer-region radius in normalised frame units")
    args = ap.parse_args()

    record = json.loads((args.run_dir / "experiment.json").read_text(encoding="utf-8"))
    pointer = (record.get("perturbation") or {}).get("pointer") or {"x": 0.5, "y": 0.5}

    ep = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)
    source = ep.frames[PERTURBED_VIEW]
    perturbed = read_video(args.run_dir / "perturbed.mp4")
    n = min(len(source), len(perturbed))
    source, perturbed = source[:n], perturbed[:n]
    if perturbed.shape[1:3] != source.shape[1:3]:
        perturbed = np.stack([cv2.resize(f, (source.shape[2], source.shape[1])) for f in perturbed])

    src_mask = target_mask(source, args.color)
    per_mask = target_mask(perturbed, args.color)

    h, w = source.shape[1:3]
    yy, xx = np.mgrid[0:h, 0:w]
    region = (((yy / h - pointer["y"]) ** 2 + ((xx / w - pointer["x"]) * (w / h) / (w / h)) ** 2)
              <= args.radius ** 2)

    src_total = float(src_mask.sum())
    per_total = float(per_mask.sum())
    src_in = float((src_mask & region).sum())
    per_in = float((per_mask & region).sum())

    result = {
        "run": args.run_dir.name,
        "color": args.color,
        "pointer": pointer,
        "frames_compared": int(n),
        "target_pixels_source_total": src_total,
        "target_pixels_perturbed_total": per_total,
        "target_pixels_source_in_pointer_region": src_in,
        "target_pixels_perturbed_in_pointer_region": per_in,
        "surviving_fraction_total": round(per_total / (src_total + 1e-9), 4),
        "surviving_fraction_in_region": round(per_in / (src_in + 1e-9), 4),
        "mean_target_pixels_per_frame_source": round(src_total / n, 1),
        "mean_target_pixels_per_frame_perturbed": round(per_total / n, 1),
    }
    surviving = result["surviving_fraction_in_region"]
    result["intervention_landed"] = bool(surviving < 0.25)
    result["interpretation"] = (
        f"{(1 - surviving) * 100:.0f}% of the target's coloured pixels inside the pointer "
        f"region were removed. "
        + (
            "The intervention landed: a null behavioural result is about the policy."
            if surviving < 0.25
            else "The intervention did NOT clearly land: a null behavioural result is "
                 "about the prompt, not the policy, and must not be reported as a "
                 "finding about the policy."
        )
    )

    (args.run_dir / "removal_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    idx = np.linspace(0, n - 1, 5).astype(int)
    strip = np.concatenate(
        [
            np.concatenate([cv2.resize(source[i], (256, 144)) for i in idx], axis=1),
            np.concatenate([cv2.resize(perturbed[i], (256, 144)) for i in idx], axis=1),
        ],
        axis=0,
    )
    cv2.imwrite(str(args.run_dir / "removal_check.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
