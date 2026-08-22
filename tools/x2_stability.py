"""Quantify X2's run-to-run variance at a fixed prompt.

X2 exposes no seed (verified absent from its OpenAPI schema). That is not a footnote:
it means the same fault at the same strength is a *different* perturbation every time
it is run, and a margin reported from one run is a sample, not a threshold.

This was first noticed accidentally - `specular_floor` at strength 1.0 passed the SEAM
gate on one run (temporal ratio 1.92) and was rejected on another (4.14). This tool
measures it deliberately: run the identical rung K times and report the spread of the
gate statistics and the perceptual distance.

The policy is never invoked, so this costs only X2 session time.

    python tools/x2_stability.py --fault specular_floor --strength 1.0 --repeats 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402
from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402
from penumbra.episodes.types import PERTURBED_VIEW  # noqa: E402
from penumbra.perturbation.spec import load_fault  # noqa: E402
from penumbra.perturbation.x2 import X2Perturbation  # noqa: E402
from penumbra.validation.seam import SeamGate, perceptual_distance  # noqa: E402


def _spread(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": round(float(a.mean()), 5),
        "std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 5),
        "min": round(float(a.min()), 5),
        "max": round(float(a.max()), 5),
        "range_over_mean": round(float((a.max() - a.min()) / (a.mean() + 1e-9)), 3),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", default="specular_floor")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    args = ap.parse_args()

    fault = load_fault(args.fault)
    ep = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)
    source = ep.frames[PERTURBED_VIEW]
    gate = SeamGate(
        max_geometry_drift=fault.max_geometry_drift,
        min_structure_retained=fault.min_structure_retained,
    )
    perturber = X2Perturbation()

    rows = []
    for i in range(args.repeats):
        t0 = time.time()
        result = await perturber.apply(ep, fault, args.strength, run_id=f"stab{i}")
        frames = result.episode.frames[PERTURBED_VIEW]
        report = gate.validate(source, frames)
        distance = perceptual_distance(source, frames)
        rows.append(
            {
                "run": i,
                "session_id": result.session_id,
                "valid": report.valid,
                "reasons": report.reasons,
                "geometry_drift": round(report.geometry_drift, 5),
                "geometry_drift_p90": round(report.geometry_drift_p90, 5),
                "structure_retained": round(report.structure_retained, 4),
                "temporal_ratio": round(report.temporal_ratio, 4),
                "rmse": round(distance["rmse"], 5),
                "ssim": round(distance["ssim"], 4),
                "received_frames": result.received_frames,
                "alignment_offset": result.alignment_offset,
                "alignment_score": round(result.alignment_score, 4),
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(
            f"run {i}: {'VALID' if report.valid else 'REJECTED'}  "
            f"drift={report.geometry_drift:.5f}  structure={report.structure_retained:.3f}  "
            f"temporal={report.temporal_ratio:.2f}  rmse={distance['rmse']:.4f}",
            flush=True,
        )
        if report.reasons:
            for r in report.reasons:
                print(f"    ! {r}", flush=True)

    valid = [r for r in rows if r["valid"]]
    record = {
        "fault": fault.name,
        "fault_version": fault.version,
        "strength": args.strength,
        "prompt": fault.rung_at(args.strength).prompt,
        "episode": ep.episode_id,
        "repeats": args.repeats,
        "model": "xmax/x2",
        "seed_available": False,
        "gate_pass_rate": round(len(valid) / len(rows), 3),
        "spread": {
            "geometry_drift": _spread([r["geometry_drift"] for r in rows]),
            "structure_retained": _spread([r["structure_retained"] for r in rows]),
            "temporal_ratio": _spread([r["temporal_ratio"] for r in rows]),
            "rmse": _spread([r["rmse"] for r in rows]),
        },
        "runs": rows,
    }
    out = RUNS_DIR / "_probe"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"x2_stability_{fault.name}_{args.strength:g}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\nSPREAD OVER IDENTICAL RUNS (same prompt, no seed available)")
    for key, stats in record["spread"].items():
        print(f"  {key:<20} mean={stats['mean']:<9} std={stats['std']:<9} "
              f"min={stats['min']:<9} max={stats['max']:<9} range/mean={stats['range_over_mean']}")
    print(f"\ngate pass rate: {record['gate_pass_rate']:.0%} of {args.repeats} identical runs")
    print(f"saved -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
