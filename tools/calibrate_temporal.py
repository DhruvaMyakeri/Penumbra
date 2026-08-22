"""Can the gate tell added dynamics apart from generative flicker? Measure, don't assume.

Looking at a finished suite's renders by eye turned up something the numbers had hidden:
the *most convincing* generations - steam rising, soap foam, a granite benchtop, a
checkered tabletop - were all rejected, while a render that replaced the robot arm with
a toy passed. Every one of those rejections fired on the same criterion, the temporal
motion ratio, and four of the five were marginal (2.57-2.80 against a 2.50 bar).

That criterion exists to catch **generative flicker**: a stream that boils and
re-hallucinates detail frame to frame, carrying motion the recording never had. But
steam legitimately moves. Foam legitimately swells. Those are the situations most worth
testing a policy against, and a criterion that structurally cannot admit them is a
capability limit, not a validity check.

The two cases differ in *where* the motion is. Flicker is everywhere; added dynamics sit
in the region that changed. So the candidate statistic is the temporal ratio restricted
to pixels whose colour did NOT change:

    residual_temporal = motion(perturbed, outside edit) / motion(source, outside edit)

near 1.0 means the underlying scene is as steady as the recording and the extra motion
belongs to the added content; well above 1.0 means the whole frame is boiling.

This tool calibrates that statistic on synthetic streams where the answer is known by
construction, then reports it for real renders. **It changes no threshold.** Nothing
here is applied to a stored verdict: a gate must not be re-tuned against results that
have already been seen, so any threshold this justifies has to be validated on a fresh
run before it decides anything.

    .venv/Scripts/python tools/calibrate_temporal.py --run runs/garage-ep0-...
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

COLOUR_CHANGE = 26.0
WORK_W = 320


def temporal_stats(source: np.ndarray, perturbed: np.ndarray, samples: int = 12) -> dict:
    """Whole-frame temporal ratio, and the same ratio outside the edited region."""
    n = min(len(source), len(perturbed)) - 1
    idx = np.unique(np.linspace(0, n - 1, min(samples, max(1, n))).astype(int))
    whole_s, whole_p, out_s, out_p, area = [], [], [], [], []
    for i in idx:
        h, w = source.shape[1:3]
        th = int(round(h * WORK_W / w))

        def prep(stack, k):
            return _gray(cv2.resize(stack[k], (WORK_W, th), interpolation=cv2.INTER_AREA))

        sa, sb = prep(source, i), prep(source, i + 1)
        pa, pb = prep(perturbed, i), prep(perturbed, i + 1)
        fs, fp = _flow_magnitude(sa, sb), _flow_magnitude(pa, pb)

        a = cv2.resize(source[i], (WORK_W, th), interpolation=cv2.INTER_AREA)
        b = cv2.resize(perturbed[i], (WORK_W, th), interpolation=cv2.INTER_AREA)
        delta = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
        changed = cv2.morphologyEx((delta > COLOUR_CHANGE).astype(np.uint8),
                                   cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        area.append(float(changed.mean()))
        whole_s.append(float(np.median(fs)))
        whole_p.append(float(np.median(fp)))
        if (~changed).sum() > 64:
            out_s.append(float(np.median(fs[~changed])))
            out_p.append(float(np.median(fp[~changed])))
    whole = float(np.mean(whole_p)) / (float(np.mean(whole_s)) + 1e-6)
    residual = (float(np.mean(out_p)) / (float(np.mean(out_s)) + 1e-6)) if out_s else float("nan")
    return {"temporal_ratio": round(whole, 4),
            "residual_temporal_ratio": round(residual, 4),
            "edited_area": round(float(np.mean(area)), 4)}


# -- synthetic streams where the answer is known by construction ---------------

def added_dynamics(frames: np.ndarray, area: float = 0.25, seed: int = 0) -> np.ndarray:
    """Animated content confined to one region. The scene underneath is untouched."""
    rng = np.random.default_rng(seed)
    out = frames.copy()
    h, w = frames.shape[1:3]
    side = float(np.sqrt(area))
    rh, rw = int(h * side), int(w * side)
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    for i in range(len(out)):
        blob = rng.normal(200, 40, (rh // 8 + 1, rw // 8 + 1, 3))
        blob = cv2.resize(blob, (rw, rh), interpolation=cv2.INTER_CUBIC)
        region = out[i, y0:y0 + rh, x0:x0 + rw].astype(np.float32)
        out[i, y0:y0 + rh, x0:x0 + rw] = np.clip(0.35 * region + 0.65 * blob, 0, 255)
    return out


def flicker(frames: np.ndarray, sigma: float = 14.0, seed: int = 0) -> np.ndarray:
    """Independent noise on every frame, everywhere. The classic generative boil."""
    rng = np.random.default_rng(seed)
    out = frames.astype(np.float32) + rng.normal(0, sigma, frames.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def shuffled(frames: np.ndarray, seed: int = 0) -> np.ndarray:
    """Temporal order destroyed - the most extreme incoherence available."""
    rng = np.random.default_rng(seed)
    return frames[rng.permutation(len(frames))]


def gamma(frames: np.ndarray, g: float = 0.5) -> np.ndarray:
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return lut[frames]


def read_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(fr) if fr else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--run", type=Path, default=None,
                    help="also report the statistic for this run's real renders")
    args = ap.parse_args()

    ep = load_episode([r for r in list_episodes() if r.episode_index == args.episode][0]
                      ).window(args.start, args.frames)
    src = ep.frames[PERTURBED_VIEW]

    legit = {"identity": src, "gamma_0.5": gamma(src)}
    # A finer area ladder, because the previous three-point version could not tell
    # "the statistic fails" from "the statistic fails once the edit swallows the frame".
    for a in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        legit["added_dynamics_%dpct" % int(a * 100)] = added_dynamics(src, a)
    bad = {
        "flicker_sigma_6": flicker(src, 6.0),
        "flicker_sigma_14": flicker(src, 14.0),
        "flicker_sigma_28": flicker(src, 28.0),
        "shuffled_frames": shuffled(src),
    }

    print(f"{'stream':<28}{'temporal':<12}{'residual':<12}{'edited area'}")
    print("-" * 66)
    result = {"legitimate": {}, "incoherent": {}, "real_renders": {}}
    for label, group, key in (("COHERENT - motion belongs to added content", legit, "legitimate"),
                              ("INCOHERENT - the whole frame boils", bad, "incoherent")):
        print("\n" + label)
        for name, frames in group.items():
            st = temporal_stats(src, frames)
            result[key][name] = st
            print(f"  {name:<26}{st['temporal_ratio']:<12.3f}"
                  f"{st['residual_temporal_ratio']:<12.3f}{st['edited_area']:.3f}")

    worst_legit = max(v["residual_temporal_ratio"] for v in result["legitimate"].values())
    best_bad = min(v["residual_temporal_ratio"] for v in result["incoherent"].values())
    separates = worst_legit < best_bad
    result["separation"] = {
        "worst_coherent_residual": round(worst_legit, 4),
        "smallest_incoherent_residual": round(best_bad, 4),
        "separates": bool(separates),
        "suggested_threshold": round(float(np.sqrt(max(worst_legit, 1e-6) * best_bad)), 4)
        if separates else None,
    }

    if args.run:
        state = json.loads((args.run / "state.json").read_text(encoding="utf-8"))
        views = state.get("views") or [PERTURBED_VIEW]
        print("\nREAL RENDERS")
        print(f"  {'situation':<28}{'temporal':<12}{'residual':<12}{'gate verdict'}")
        for s in state["scenarios"]:
            v = args.run / f"media/{s['name']}.mp4"
            if not v.exists():
                continue
            stack = read_video(v)
            if stack is None:
                continue
            h, w = stack.shape[1] // 2, stack.shape[2] // max(1, len(views))
            per = stack[:, h:, :w]
            n = min(len(per), len(src))
            st = temporal_stats(src[:n], per[:n])
            result["real_renders"][s["name"]] = {**st, "status": s["status"]}
            print(f"  {s['name']:<28}{st['temporal_ratio']:<12.3f}"
                  f"{st['residual_temporal_ratio']:<12.3f}{s['status']}")

    out = RUNS_DIR / "_probe" / "gate_calibration_temporal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "-" * 66)
    s = result["separation"]
    print(f"coherent residual <= {s['worst_coherent_residual']:.3f}   "
          f"incoherent residual >= {s['smallest_incoherent_residual']:.3f}   "
          f"separates: {s['separates']}")
    if s["suggested_threshold"]:
        print(f"a threshold would sit near {s['suggested_threshold']:.3f} - but it changes "
              f"nothing until a FRESH run validates it.")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
