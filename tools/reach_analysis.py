"""Can X2 reach the distance at which this policy is detectably affected?

This is the question the whole project turns on, reduced to two numbers that can be
measured independently:

  DETECTION THRESHOLD   the perceptual distance at which a perturbation of this camera
                        view produces a statistically detectable shift in the policy's
                        action distribution. Measured with the blackout positive
                        control, which is not a fault - it is a ruler.

  ADMISSIBLE REACH      the largest perceptual distance X2 has produced while still
                        passing the SEAM validity gate. Measured across every real run.

If ADMISSIBLE REACH < DETECTION THRESHOLD, the thesis fails on this policy for a
concrete reason: the generative operator cannot get far enough from the source without
corrupting the scene, so no admissible generative perturbation could ever register.
That is a real, publishable negative result rather than an unexplained absence of
findings.

If they overlap, the search should be pointed at that band.

    python tools/reach_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402


def collect_generative() -> list[dict]:
    """Every X2 perturbation ever run, with its distance, gate verdict and p-value."""
    rows: list[dict] = []
    for d in sorted(RUNS_DIR.glob("*/experiment.json")):
        try:
            rec = json.loads(d.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        fault = (rec.get("perturbation") or {}).get("name", "?")
        for e in (rec.get("divergence") or {}).get("evaluations", []):
            pert = e.get("perturbation") or {}
            if pert.get("model") != "xmax/x2":
                continue
            g = e.get("group_test") or {}
            rows.append({
                "run": d.parent.name,
                "fault": fault,
                "strength": e.get("strength"),
                "rmse": (e.get("distance") or {}).get("rmse"),
                "gate": (e.get("validation") or {}).get("status"),
                "temporal_ratio": (e.get("validation") or {}).get("temporal_ratio"),
                "p_joint": g.get("p_value"),
                "significant": g.get("any_significant", g.get("significant_at_alpha")),
            })
    return rows


def collect_ruler() -> list[dict]:
    """The blackout sensitivity ladder: distance -> detectability."""
    rows: list[dict] = []
    seen: set[float] = set()
    for path in sorted((RUNS_DIR / "_probe").glob("positive_control*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        for r in rec.get("results", []):
            g = r.get("group_test") or {}
            if r.get("strength") in seen:
                continue
            seen.add(r.get("strength"))
            rows.append({
                "strength": r.get("strength"),
                "rmse": (r.get("distance") or {}).get("rmse"),
                "p_joint": g.get("p_value"),
                "p_gripper": g.get("gripper_p_value"),
                "effect_size": g.get("effect_size_cohens_d"),
                "significant": g.get("any_significant", g.get("significant_at_alpha")),
            })
    return sorted(rows, key=lambda r: (r["rmse"] is None, r["rmse"]))


def main() -> None:
    gen = collect_generative()
    ruler = collect_ruler()

    print("=" * 78)
    print("DETECTION THRESHOLD  (blackout ruler: how big must a change be to register?)")
    print("=" * 78)
    if not ruler:
        print("  no positive-control data yet - run tools/positive_control.py")
    for r in ruler:
        mark = "DETECTED" if r["significant"] else "not detected"
        pg = f"{r['p_gripper']:.4f}" if r.get("p_gripper") is not None else "n/a"
        print(f"  blackout {r['strength']:>5}  rmse={r['rmse']:.4f}  "
              f"p_joint={r['p_joint']:.4f}  p_grip={pg}  d={r['effect_size']:+.2f}  {mark}")

    detected = [r for r in ruler if r["significant"] and r["rmse"] is not None]
    not_detected = [r for r in ruler if not r["significant"] and r["rmse"] is not None]
    threshold_lo = max((r["rmse"] for r in not_detected), default=None)
    threshold_hi = min((r["rmse"] for r in detected), default=None)

    print()
    print("=" * 78)
    print("ADMISSIBLE REACH  (how far can X2 get while passing the validity gate?)")
    print("=" * 78)
    for r in sorted(gen, key=lambda x: (x["rmse"] is None, x["rmse"])):
        p = f"{r['p_joint']:.4f}" if r.get("p_joint") is not None else "n/a"
        print(f"  {r['fault']:<16} s={r['strength']:<5} rmse={r['rmse']:.4f}  "
              f"gate={r['gate']:<9} temporal={r['temporal_ratio']:.2f}  p={p}")

    valid = [r for r in gen if r["gate"] == "VALID" and r["rmse"] is not None]
    reach = max((r["rmse"] for r in valid), default=None)
    rejected = [r for r in gen if r["gate"] == "REJECTED" and r["rmse"] is not None]
    reach_rejected = max((r["rmse"] for r in rejected), default=None)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  admissible reach (max RMSE, gate VALID)      : "
          f"{reach:.4f}" if reach is not None else "  admissible reach: no valid runs yet")
    if reach_rejected is not None:
        print(f"  max RMSE attempted but gate-REJECTED         : {reach_rejected:.4f}")
    if threshold_lo is not None:
        print(f"  largest distance NOT detected                : {threshold_lo:.4f}")
    if threshold_hi is not None:
        print(f"  smallest distance DETECTED                   : {threshold_hi:.4f}")

    verdict = "INCONCLUSIVE - not enough ruler points yet"
    if reach is not None and threshold_hi is not None:
        if reach < threshold_hi:
            gap = threshold_hi / reach
            verdict = (
                f"X2's admissible reach ({reach:.4f}) falls SHORT of the smallest "
                f"detected distance ({threshold_hi:.4f}) by {gap:.1f}x. On this policy "
                f"and this view, no admissible generative perturbation tested so far "
                f"gets far enough from the source to register. The nulls have a "
                f"mechanical explanation, not a mysterious one."
            )
        else:
            verdict = (
                f"X2's admissible reach ({reach:.4f}) EXCEEDS the smallest detected "
                f"distance ({threshold_hi:.4f}). Detectable, admissible generative "
                f"perturbations are possible - point the search at that band."
            )
    print()
    for line in verdict.split(". "):
        if line.strip():
            print(f"  {line.strip()}{'.' if not line.endswith('.') else ''}")

    out = {
        "detection_ruler": ruler,
        "generative_runs": gen,
        "admissible_reach_rmse": reach,
        "max_rejected_rmse": reach_rejected,
        "largest_not_detected_rmse": threshold_lo,
        "smallest_detected_rmse": threshold_hi,
        "verdict": verdict,
    }
    path = RUNS_DIR / "_probe" / "reach_analysis.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
