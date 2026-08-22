"""Calibrate SEAM's thresholds against real data instead of guessing them.

Measures the flow statistic for transformations we KNOW are appearance-only
(identity, brightness, a real X2 output) and for transformations we KNOW moved the
geometry (pixel shifts of known size), then reports where a threshold can separate
them. The research document's suggested 0.05 was an order of magnitude too lenient:
on 640-wide DROID footage it only rejects shifts larger than ~36 pixels.

Writes runs/_probe/gate_calibration.json.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penumbra.config import RUNS_DIR
from penumbra.episodes.droid import list_episodes, load_episode
from penumbra.validation.seam import SeamGate, _flow_magnitude

def drift(gate, a, b):
    idx = np.unique(np.linspace(0, len(a) - 1, 12).astype(int))
    A, B = gate._prep(a, idx), gate._prep(b, idx)
    diag = float(np.hypot(*A[0].shape))
    fl = [_flow_magnitude(x, y) for x, y in zip(A, B)]
    return (float(np.median([np.median(f) for f in fl])) / diag,
            float(np.median([np.percentile(f, 90) for f in fl])) / diag)

def main():
    ep = load_episode([r for r in list_episodes() if r.episode_index == 0][0]).window(120, 96)
    src = ep.frames["exterior_image_1_left"]
    gate = SeamGate()
    good, bad = {}, {}
    good["identity"] = drift(gate, src, src)
    good["brightness_1.5x+40"] = drift(gate, src, np.clip(src.astype(np.float32) * 1.5 + 40, 0, 255).astype(np.uint8))
    good["gamma_0.5"] = drift(gate, src, cv2.LUT(src, (np.linspace(0,1,256) ** 0.5 * 255).astype(np.uint8)))
    for d in sorted(glob.glob(str(RUNS_DIR / "perturb-*")) + glob.glob(str(RUNS_DIR / "search-*"))):
        p = Path(d) / "perturbed.mp4"
        if not p.exists():
            continue
        cap = cv2.VideoCapture(str(p)); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(fr) == len(src):
            good[f"x2::{Path(d).name}"] = drift(gate, src, np.stack(fr))
    for px in (2, 4, 8, 16, 32):
        bad[f"shift_{px}px_of_{src.shape[2]}"] = drift(gate, src, np.roll(src, px, axis=2))

    worst_good = max(v[0] for v in good.values())
    best_bad = min(v[0] for v in bad.values())
    worst_good90 = max(v[1] for v in good.values())
    best_bad90 = min(v[1] for v in bad.values())
    rec = {
        "appearance_only": {k: {"median": round(v[0], 5), "p90": round(v[1], 5)} for k, v in good.items()},
        "geometry_moved": {k: {"median": round(v[0], 5), "p90": round(v[1], 5)} for k, v in bad.items()},
        "separation": {
            "worst_appearance_median": round(worst_good, 5),
            "smallest_geometric_median": round(best_bad, 5),
            "worst_appearance_p90": round(worst_good90, 5),
            "smallest_geometric_p90": round(best_bad90, 5),
        },
        "recommended": {
            "max_geometry_drift": round(float(np.sqrt(max(worst_good, 1e-6) * best_bad)), 5),
            "max_geometry_drift_p90": round(float(np.sqrt(max(worst_good90, 1e-6) * best_bad90)), 5),
        },
        "note": "thresholds are the geometric mean of the worst legitimate appearance "
                "change and the smallest geometric corruption, i.e. equidistant in log space",
    }
    out = RUNS_DIR / "_probe"; out.mkdir(parents=True, exist_ok=True)
    (out / "gate_calibration.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))

main()
